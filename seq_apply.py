import os
import json

import hydra
import tqdm
import torch
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
import sklearn.metrics as skm

from Transforms import VideoNormalize
from dataloader.seq_dataset import SyntaxDataset
from models.seq_model import SyntaxLightningModule

@hydra.main(version_base=None,
            config_path="configs",
            config_name="seq_apply")
def main(
        cfg: DictConfig
):
    output_dir = HydraConfig.get().runtime.output_dir

    # Run the folds for each side independently
    left_bin_prob, left_bin, left_syntax, left_sids = run_for_artery(
        cfg,
        "left",
        cfg.model.model_path_left
    )

    right_bin_prob, right_bin, right_syntax, right_sids = run_for_artery(
        cfg,
        "right",
        cfg.model.model_path_right
    )

    # Check if the results are same length & are overlapping
    assert len(left_sids) == len(right_sids)
    assert left_sids == right_sids

    # Init output paths
    src_path = f"rnn_folds/step2_rnn_fold{cfg.dataset.fold:02d}_{cfg.test_data}.json"
    full_src_path = os.path.join(cfg.dataset.root, src_path)
    dst_path = f"{cfg.dataset.fold:02d}_{cfg.test_data}_results.json"
    full_dst_path = os.path.join(output_dir, dst_path)

    # Load inital dataset with labels
    dataset = json.load(open(full_src_path))

    # Check if everything fits together
    assert len(dataset) == len(left_sids) == len(right_sids)

    syntax_true = []
    syntax_pred = []
    # Combine all results into the original dataset as "results" object
    for rec, sid, l_prob, l_bin, l_syntax, r_prob, r_bin, r_syntax in zip(dataset, left_sids, left_bin_prob, left_bin,
                                                                          left_syntax, right_bin_prob, right_bin,
                                                                          right_syntax):
        assert rec["study_id"] == sid
        rec["prediction"] = {
            "left_prob": l_prob,
            "left_bin": l_bin,
            "left_syntax": l_syntax,
            "right_prob": r_prob,
            "right_bin": r_bin,
            "right_syntax": r_syntax,
            "syntax": l_syntax + r_syntax,
        }
        syntax_true.append(rec["syntax"])
        syntax_pred.append(l_syntax + r_syntax)
    r2 = skm.r2_score(syntax_true, syntax_pred)
    # Test metric for seeing if the numbers make sense (note: only useful with linear scaling applied)
    print("SYNTAX R2", r2)

    # Write results
    with open(full_dst_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def run_for_artery(
        cfg,
        artery,
        model_path
):
    pl.seed_everything(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple preprocessing for inference
    if cfg.training.mvl_preprocessing:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        test_transform = T.Compose(
            [
                ToTensorVideo(),
                T.Resize(size=cfg.video.size, antialias=True),
                VideoNormalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        # Calculated on the "0" folder of the dataset
        dataset_mean = [0.55871]
        dataset_std = [0.151]

        test_transform = T.Compose(
            [
                ToTensorVideo(),
                T.Resize(size=cfg.video.size, antialias=True),
                VideoNormalize(mean=dataset_mean, std=dataset_std),
            ]
        )

    src_path = f"rnn_folds/step2_rnn_fold{cfg.dataset.fold:02d}_{cfg.test_data}.json"

    # Dataloader init
    test_set = SyntaxDataset(
        root=cfg.dataset.root,
        meta=src_path,
        train=False,
        length=cfg.video.frames_per_clip,
        label=f"syntax_{artery}",
        artery=artery,
        inference=True,
        transform=test_transform,
        duplicate_channels=cfg.training.mvl_preprocessing,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=1,  # batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model = SyntaxLightningModule(
        num_classes=cfg.dataset.num_classes,
        variant=cfg.model.variant,
        lr=1e-5,
        use_three_channel_input=cfg.training.mvl_preprocessing,
    )

    weights = torch.load(model_path, map_location=device, weights_only=False)
    if "state_dict" in weights:
        model.load_state_dict(weights["state_dict"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    Y = []
    Y_syntax = []
    P_bin_prob = []
    P_bin = []
    P_syntax = []
    sids = []

    with torch.inference_mode():
        for x, [y], [t], [_weight_], [sid] in tqdm.tqdm(test_dataloader):
            if len(x.shape) == 1:
                bin_prob = 0.0
                val_syntax = 0.0
            else:
                try:
                    x = x.to(device)
                    pred = model(x)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("OOM:", sid, x.shape)
                        torch.cuda.empty_cache()
                        continue
                    raise

                # unpack explicitly
                bin_logit, val_log = pred[0]
                bin_prob = torch.sigmoid(bin_logit).item()
                val_syntax = val_log.item() * 70
                bin = round(bin_prob)
                y_syntax = float(t)
            Y.append(y)
            Y_syntax.append(y_syntax)
            P_bin_prob.append(bin_prob)
            P_bin.append(bin)
            P_syntax.append(val_syntax)
            sids.append(sid)

    return P_bin_prob, P_bin, P_syntax, sids


if __name__ == "__main__":
    main()
