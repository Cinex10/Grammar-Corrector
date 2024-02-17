import torch
import wandb
import pandas as pd
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import hydra
from data import C4200DatasetModule
from model import T5Model



@hydra.main(config_path='config',config_name='config.yaml',version_base=None)
def main(cfg):
    data = C4200DatasetModule(cfg)
    model = T5Model(cfg)

    wandb.login()
    wandb_logger = WandbLogger(project='grammar_corrector',
                           # offline=False,
                           log_model=True)
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(model,data)
    wandb.finish()

checkpoint_callback = ModelCheckpoint(
    dirpath="./models",
    filename="best-checkpoint.ckpt",
    monitor="valid/loss",
    mode="min",
)

early_stopping_callback = EarlyStopping(
    monitor="valid/loss", patience=3, verbose=True, mode="min"
)

if __name__ == "__main__":
    main()
