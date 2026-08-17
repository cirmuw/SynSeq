import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataloader.seq_dataset import SyntaxDataset
from models.seq_model import SyntaxLightningModule
from utils import generate_transforms


@hydra.main(version_base=None,
            config_path="configs",
            config_name="seq_train")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    torch.set_float32_matmul_precision("high")
    print(cfg)

    # Initialize artery identifier
    Artery = cfg.dataset.artery.capitalize()

    # Seed everything
    pl.seed_everything(cfg.training.seed)

    # Generate appropriate channel transforms
    channels = 3 if cfg.training.mvl_preprocessing else 1
    train_transform, test_transform = generate_transforms(channels, cfg.video.size)

    # Dataloader/Dataset setup
    train_set = SyntaxDataset(
        root=cfg.dataset.root,
        meta=f"rnn_folds/{cfg.dataset.fold_prefix}_rnn_fold{int(cfg.dataset.fold):02d}_train.json",
        train=True,
        length=cfg.video.frames_per_clip,
        label=f"syntax_{cfg.dataset.artery}",
        artery=cfg.dataset.artery,
        transform=train_transform,
        healthy_weight=cfg.training.healthy_weight,
        use_log_space=cfg.training.use_log_space
    )

    val_set = SyntaxDataset(
        root=cfg.dataset.root,
        meta=f"rnn_folds/{cfg.dataset.fold_prefix}_rnn_fold{int(cfg.dataset.fold):02d}_eval.json",
        train=False,
        length=cfg.video.frames_per_clip,
        label=f"syntax_{cfg.dataset.artery}",
        artery=cfg.dataset.artery,
        transform=test_transform,
        healthy_weight=cfg.training.healthy_weight,
        use_log_space=cfg.training.use_log_space
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    # Test-print for checking if shapes are alright
    x, y, t, w, p = next(iter(train_dataloader))
    print(x.shape)
    x, y, t, w, p = next(iter(val_dataloader))
    print(x.shape)

    # Load correct backbone (3 channel/4 channel, specialized pretraining variants)
    backbone_weight = f"backbone/{cfg.dataset.artery}{'_3c' if cfg.training.mvl_preprocessing else ''}_{cfg.model.weight_path}"
    if "=" in backbone_weight.split(".")[0]:
        used_backbone = f"_backbone={backbone_weight.split('.')[0].split('=')[1]}"
    else:
        used_backbone = ""
    log_name = f"artery={Artery}_fold={int(cfg.dataset.fold):02d}_healthy-weight={cfg.training.healthy_weight}_mvl-preprocessing={cfg.training.mvl_preprocessing}_use-log-space={cfg.training.use_log_space}_clf={cfg.loss.clf}_reg={cfg.loss.reg}_rank={cfg.loss.rank}_variant={cfg.model.variant}{used_backbone}"

    print("Using backbone:", backbone_weight)

    # Train setup and run (head only)
    if cfg.training.head_only:
        model = build_model(
            cfg, x, output_dir,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            save_name="result_head_only_best",
            max_epochs=cfg.training.max_epochs,
            weight_path=backbone_weight,
            freeze_backbone=True,
        )

        run_trainer(
            model,
            output_dir,
            log_name,
            phase="head_only",
            max_epochs=cfg.training.max_epochs,
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            ckpt_name="result_head_only",
        )
    # Train setup and run (head then full)
    else:
        # ---- PRETRAIN HEAD ----
        pre_model = build_model(
            cfg, x, output_dir,
            lr=cfg.training.lr_frozen,
            weight_decay=cfg.training.weight_decay_frozen,
            save_name="result_pre_best",
            max_epochs=10,
            weight_path=backbone_weight,
            freeze_backbone=True,
        )

        run_trainer(
            pre_model,
            output_dir,
            log_name,
            phase="pre",
            max_epochs=10,
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            ckpt_name="result_pre",
        )

        pre_ckpt = f"{output_dir}/result_pre.pt"

        # ---- FINE-TUNE FULL MODEL ----
        post_model = build_model(
            cfg, x, output_dir,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            save_name="result_post_best",
            max_epochs=cfg.training.max_epochs - 10,
            pl_weight_path=pre_ckpt,
            freeze_backbone=False,
        )

        run_trainer(
            post_model,
            output_dir,
            log_name,
            phase="post",
            max_epochs=cfg.training.max_epochs - 10,
            cfg=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            ckpt_name="result_post",
        )
def run_trainer(
    model,
    output_dir: str,
    log_name: str,
    phase: str,
    max_epochs: int,
    cfg,
    train_dataloader,
    val_dataloader,
    ckpt_name: str,
):
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]

    logger = TensorBoardLogger(
        save_dir=f"{output_dir}/tb",
        name=f"{log_name}_{phase}"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=cfg.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"{output_dir}/{ckpt_name}.pt")

    return trainer


def build_model(cfg, x, output_dir, save_name, lr, weight_decay, **kwargs):
    return SyntaxLightningModule(
        num_classes=cfg.dataset.num_classes,
        video_shape=x.shape[1:],
        lr=lr,
        variant=cfg.model.variant,
        weight_decay=weight_decay,
        max_epochs=kwargs.get("max_epochs"),
        loss_clf=cfg.loss.clf,
        loss_reg=cfg.loss.reg,
        loss_rank=cfg.loss.rank,
        use_three_channel_input=cfg.training.mvl_preprocessing,
        loss_clf_weight=cfg.loss.clf_weight,
        loss_reg_weight=cfg.loss.reg_weight,
        loss_rank_weight=cfg.loss.rank_weight,
        weight_path=kwargs.get("weight_path"),
        pl_weight_path=kwargs.get("pl_weight_path"),
        freeze_backbone=kwargs.get("freeze_backbone", False),
        save_path=f"{output_dir}/{save_name}.pt",
    )

if __name__ == "__main__":
    main()
