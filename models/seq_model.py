import math
from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, optim
import lightning.pytorch as pl
import torchvision.models.video as tvmv
import sklearn.metrics as skm


class SyntaxLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float,
        variant: str, # mean, lstm_mean, lstm_last, gru_mean, gru_last, bert_mean, bert_cls
        loss_clf="BCEWithLogitsLoss", # or None
        loss_reg="MSE", # or Huber, None
        loss_rank = None, # or something different
        loss_clf_weight = 1,
        loss_reg_weight = 1,
        loss_rank_weight = 1,
        weight_decay: float = 0,
        max_epochs: int = None,
        weight_path: str = None,
        save_path: str = None,
        pl_weight_path: str = None,
        freeze_backbone = False,
        use_three_channel_input = False,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        self.save_path = save_path
        self.weight_path = weight_path
        self.variant = variant
        self.loss_clf_weight = loss_clf_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_rank_weight = loss_rank_weight
        self.best_rmse = math.inf
        # Video ResNet
        self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights.DEFAULT)

        # Replace first layer with one channel if only it is used
        if not use_three_channel_input:
            old_conv = self.model.stem[0]

            new_conv = nn.Conv3d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            with torch.no_grad():
                # Luminance weighting (e.g. averaging, but with intensity), maybe is a bit better, should not matter too much
                weights = torch.tensor([0.2989, 0.5870, 0.1140], device=old_conv.weight.device)
                new_conv.weight[:] = (old_conv.weight * weights[None, :, None, None, None]).sum(dim=1, keepdim=True)

            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

            self.model.stem[0] = new_conv


        self.lr = lr

        # Choose loss functions based on config
        if loss_clf == "BCEWithLogitsLoss":
            self.loss_clf = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_clf = None

        if loss_reg == "MSE":
            self.loss_reg = nn.MSELoss(reduction='none')
        elif loss_reg == "Huber":
            self.loss_reg = nn.HuberLoss(reduction='none', delta=0.1)
        else:
            self.loss_reg = None

        # Add prototype head that likely will be replaced downstream
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=1, bias=True)


        if weight_path is not None:
            print("Load model weights")
            weights = torch.load(weight_path, weights_only=False)
            if "state_dict" in weights:
                missing, unexpected = self.load_state_dict(weights["state_dict"], strict=False)
            else:
                missing, unexpected = self.model.load_state_dict(weights, strict=False)

            print("Missing:", missing)
            print("Unexpected:", unexpected)

        if self.variant != "mean_out":
            self.model.fc = nn.Identity()

        if self.variant == "mean_out":
            pass # self.model only
        elif self.variant in ("gru_mean", "gru_last"): # GRU with FC projector
            self.rnn = nn.GRU(in_features, in_features//4, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features=in_features//4, out_features=num_classes, bias=True)
        elif self.variant in ("lstm_mean", "lstm_last"): # LSTM with internal projector
            self.lstm = nn.LSTM(
                input_size=in_features, hidden_size=in_features//4, proj_size=num_classes, batch_first=True
            )
        elif self.variant == "lstm_mean_2head":
            self.lstm1 = nn.LSTM(
                input_size=in_features, hidden_size=in_features//4, proj_size=1, batch_first=True
            )
            self.lstm2 = nn.LSTM(
                input_size=in_features, hidden_size=in_features // 4, proj_size=1, batch_first=True
            )
        elif self.variant == "mean": # Mean of embeddings
            self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        elif self.variant in ("bert_mean", "bert_cls"):# Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_features, nhead=4, batch_first=True, dim_feedforward=in_features * 2
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.dropout = nn.Dropout(0.05)
            self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        else:
            raise ValueError(f"Unknown model variant {self.variant}")

        if pl_weight_path is not None:
            print("Load LightningModule weights")
            pl_state_dict = torch.load(pl_weight_path)["state_dict"]
            self.load_weights(pl_state_dict, self.model, "model")
            if self.variant == "mean_out":
                pass # self.model only
            elif self.variant in ("gru_mean", "gru_last"):
                self.load_weights(pl_state_dict, self.rnn, "rnn")
                self.load_weights(pl_state_dict, self.fc, "fc")
            elif self.variant in ("lstm_mean", "lstm_last"):
                self.load_weights(pl_state_dict, self.lstm, "lstm")
            elif self.variant == "lstm_mean_2head":
                self.load_weights(pl_state_dict, self.lstm1, "lstm1")
                self.load_weights(pl_state_dict, self.lstm2, "lstm2")
            elif self.variant == "mean":
                self.load_weights(pl_state_dict, self.fc, "fc")
            elif self.variant in ("bert_mean", "bert_cls"):
                self.load_weights(pl_state_dict, self.encoder, "encoder")
                self.load_weights(pl_state_dict, self.fc, "fc")
            else:
                raise ValueError(f"Unknown model variant {self.variant}")

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        self.y_val = []
        self.p_val = []
        self.r_val = []
        self.ty_val = []
        self.tp_val = []

        if freeze_backbone:
            # Freeze backbone
            for param in self.model.parameters():
                param.requires_grad = False

    def load_weights(self, pl_state_dict, model, prefix):
        model_state_dict = {}
        for key, value in pl_state_dict.items():
            if key.startswith(f"{prefix}."):
                new_key = key.split(".", 1)[1]
                assert new_key not in model_state_dict
                model_state_dict[new_key] = value
        model.load_state_dict(model_state_dict)


    def forward(self, x):
        batch_seq_shape = x.shape[0:2]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.model(x)
        x = torch.unflatten(x, 0, batch_seq_shape) # (batch, seq) if batch_first=True

        if self.variant == "mean_out":  # Mean of outputs
            x = torch.mean(x, dim=1)
        elif self.variant in ("gru_mean", "gru_last"):
            _all_outs_, [_last_out_] = self.rnn(x)
            if self.variant == "gru_mean":
                x = torch.mean(_all_outs_, dim=1) # mean of all outs
            else: # "gru_last"
                x = _last_out_ # last out
            x = self.dropout(x)
            x = self.fc(x)
        elif self.variant in ("lstm_mean", "lstm_last"):
            _all_outs_, (_last_out_, _last_state_) = self.lstm(x)
            if self.variant == "lstm_mean":
                x = torch.mean(_all_outs_, dim=1) # mean of all outs
            else: # "lstm_last"
                x = _last_out_ # last out
        elif self.variant == "lstm_mean_2head":
            _all_outs_1, (_last_out_, _last_state_) = self.lstm1(x)
            _all_outs_2, (_last_out_, _last_state_) = self.lstm2(x)
            x = torch.cat([torch.mean(_all_outs_1, dim=1), torch.mean(_all_outs_2, dim=1)], dim=1)
        elif self.variant == "mean":
            x = torch.mean(x, dim=1)
            x = self.fc(x)
        elif self.variant in ("bert_mean", "bert_cls"):
            if self.variant == "bert_cls":
                x = F.pad(x, (0,0,1,0), "constant", 0) # add first CLS token
            x = self.encoder(x)
            if self.variant == "bert_mean":
                x = torch.mean(x, dim=1) # mean of all tokens
            else: # "bert_cls"
                x = x[:,0,:] # CLS token output
            x = self.dropout(x)
            x = self.fc(x)
        else:
            raise ValueError(f"Unknown model variant {self.variant}")

        return x

    def training_step(self, batch, batch_idx):
        x, y, target, sample_weight, path = batch
        y_hat = self(x)
        yp_clf = y_hat[:,0:1]
        yp_reg = y_hat[:,1:]
        loss = 0

        if self.loss_clf is not None:
            clf_loss = self.loss_clf(yp_clf, y)
            clf_loss = clf_loss.mean()
            loss += clf_loss * self.loss_clf_weight
            self.log("clf_loss", clf_loss, prog_bar=True)

        if self.loss_reg is not None:
            reg_loss = self.loss_reg(yp_reg, target)
            reg_loss = reg_loss * sample_weight
            reg_loss = reg_loss.mean()
            loss += reg_loss * self.loss_reg_weight
            self.log("reg_loss", reg_loss, prog_bar=True)

        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, target, sample_weight, path = batch
        y_hat = self(x)
        yp_clf = y_hat[:,0:1]
        yp_reg = y_hat[:,1:]

        clf_loss = 0
        reg_loss = 0

        if self.loss_clf is not None:
            clf_loss = self.loss_clf(yp_clf, y)
            clf_loss = clf_loss * sample_weight
            clf_loss = clf_loss.mean()

        if self.loss_reg is not None:
            reg_loss = self.loss_reg(yp_reg, target)
            reg_loss = reg_loss * sample_weight
            reg_loss = reg_loss.mean()

        loss = clf_loss + reg_loss
        y_pred = torch.sigmoid(yp_clf)

        self.y_val.append(int(y[...,0].cpu()))
        self.p_val.append(float(y_pred[...,0].cpu()))
        self.r_val.append(round(float(y_pred[...,0].cpu())))

        self.ty_val.append(float(target[...,0].cpu()))
        self.tp_val.append(float(yp_reg[...,0].cpu()))

        return loss

    def on_validation_epoch_end(self):
        try:
            auc = skm.roc_auc_score(self.y_val, self.p_val)
            f1 = skm.f1_score(self.y_val, self.r_val)
            acc = skm.accuracy_score(self.y_val, self.r_val)
            self.log("val_clf_auc", auc, prog_bar=True)
            self.log("val_clf_f1", f1, prog_bar=True)
            self.log("val_clf_acc", acc, prog_bar=True)

            rmse = skm.root_mean_squared_error(self.ty_val, self.tp_val)
            self.log("val_rmse", rmse, prog_bar=True)

            if self.save_path and rmse < self.best_rmse:
                torch.save(self.state_dict(), self.save_path)
                self.best_rmse = rmse
        except ValueError as err:
            print("Skipped: Validation score calculation due to error")
            print(err)
        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        self.ty_val.clear()
        self.tp_val.clear()

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.weight_path:    # pretrain without video backbone
            if self.variant == "mean_out":
                ms = [self.model.fc]
            elif self.variant in ("gru_mean", "gru_last"):
                ms = [self.rnn, self.fc]
            elif self.variant in ("lstm_mean", "lstm_last"):
                ms = [self.lstm]
            elif self.variant == "lstm_mean_2head":
                ms = [self.lstm1, self.lstm2]
            elif self.variant == "mean": # Mean of embeddings
                ms = [self.fc] # mean
            elif self.variant in ("bert_mean", "bert_cls"): # transformer encoder
                ms = [self.encoder, self.fc] # transformer
            
            params = []
            for m in ms:
                for p in m.parameters():
                    params.append(p)
        else:
            params = self.parameters()

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, target, sample_weight, path = batch
        y_hat = self(x)
        yp_clf = y_hat[:,0:1]
        y_pred = torch.sigmoid(yp_clf)

        return {"y": y, "y_pred": torch.round(y_pred), "y_prob": y_pred}
