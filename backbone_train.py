import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader


from dataloader.backbone_dataset import SyntaxDataset
from models.backbone_model import SyntaxLightningModule
from utils import generate_transforms


@hydra.main(version_base=None,
            config_path="configs",
            config_name="backbone_train")
def main(
    cfg: DictConfig
):
    torch.set_float32_matmul_precision("high")
    # Print config for transparency
    print(cfg)
    # Initialize artery identifier vars
    Artery = cfg.dataset.artery.capitalize()
    if cfg.dataset.artery == "left":
        artery_bin = 0
    elif cfg.dataset.artery == "right":
        artery_bin = 1
    else:
        raise ValueError(f"Unknown artery '{cfg.dataset.artery}'")

    # Seed everything
    pl.seed_everything(cfg.training.seed)

    # Generate appropriate channel transforms
    channels = 3 if cfg.training.mvl_preprocessing else 1
    channel_prefix= "_3c" if channels == 3 else ""
    train_transform, test_transform = generate_transforms(channels, cfg.video.size)

    # Load datasets
    train_set = SyntaxDataset(
        root=cfg.dataset.root,
        meta = f"folds/{cfg.dataset.fold_prefix}_fold{cfg.dataset.fold:02d}_train.json",
        train = True,
        length = cfg.video.frames_per_clip,
        label = f"syntax_{cfg.dataset.artery}",
        artery_bin=artery_bin,
        transform=train_transform,
    )

    val_set = SyntaxDataset(
        root=cfg.dataset.root,
        meta = f"folds/{cfg.dataset.fold_prefix}_fold{cfg.dataset.fold:02d}_eval.json",
        train = False,
        length = cfg.video.frames_per_clip,
        label = f"syntax_{cfg.dataset.artery}",
        artery_bin=artery_bin,
        transform=test_transform,
    )

    # Init dataloaders
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
        batch_size=1, #batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    # Test-print for checking if shapes are alright
    x, y, w, p = next(iter(train_dataloader))
    print(x.shape)
    
    # Train last fc layer
    model = SyntaxLightningModule(
        num_classes=1,
        lr=1e-4,
        weight_decay=0.001,
        max_epochs=10,
        use_three_channel_input=channels == 3,
        save_path=f"backbone/{cfg.dataset.artery}{channel_prefix}_pre_fold{cfg.dataset.fold:02d}.pt"
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("back_logs", name=f"{Artery}BinSyntax_R3D_pre_fold{cfg.dataset.fold:02d}")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        fast_dev_run=cfg.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"backbone/{Artery}BinSyntax_R3D_pre_fold{cfg.dataset.fold:02d}.pt")

    # Train all layers
    model = SyntaxLightningModule(
        num_classes=1,
        lr=1e-4,
        weight_decay=0.001,
        max_epochs=cfg.training.max_epochs,
        use_three_channel_input=channels==3,
        weight_path=f"backbone/{cfg.dataset.artery}{channel_prefix}_pre_fold{cfg.dataset.fold:02d}.pt",
        save_path=f"backbone/{cfg.dataset.artery}{channel_prefix}_post_fold{cfg.dataset.fold:02d}.pt"
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("back_logs", name=f"{Artery}BinSyntax_R3D_full_fold{cfg.dataset.fold:02d}")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        fast_dev_run=cfg.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"backbone/{Artery}BinSyntax_R3D_full_fold{cfg.dataset.fold:02d}.pt")


if __name__ == "__main__":
    main()
