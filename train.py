import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.datareader import TinyShakespeare
from model.transformer_experiment import Transformer_Experiment


def main():
    parser = argparse.ArgumentParser(description="Generic runner for Transformer")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="config.yaml",
    )
    args = parser.parse_args()
    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        print("Debugging Mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        config["num_workers"] = 0

    data = Path(config["data_path"])

    train_reader = TinyShakespeare(
        data, version="train", block_size=config["block_size"]
    )
    train_dataloader = DataLoader(
        train_reader,
        shuffle=True,
        batch_size=config["batch_size"],
        drop_last=False,
        num_workers=12,
    )
    val_reader = TinyShakespeare(data, version="val", block_size=config["block_size"])
    val_dataloader = DataLoader(
        val_reader,
        shuffle=False,
        batch_size=config["batch_size"],
        drop_last=False,
        num_workers=12,
    )

    model = Transformer_Experiment(
        vocab_size=train_reader.vocab_size,
        embedding_dim=config["embedding_dim"],
        block_size=train_reader.block_size,
        num_heads=config["num_heads"],
        num_blocks=config["num_blocks"],
        dropout=0.5,
    )

    loss_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=4,
        mode="min",
        filename="model-{epoch:02d}-{val_loss:.6f}",
    )

    callbacks = [
        loss_callback,
    ]

    version_path = f"LM-Transformer-{datetime.now().strftime('%d-%m_%H:%M:%S')}"

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=Path("."),
        version=version_path,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=30,
        log_every_n_steps=1,
    )

    pprint(config)
    for key, value in config.items():
        trainer.logger.experiment.add_text(key, str(value), global_step=0)

    log_path = Path(tb_logger.log_dir)
    with open(log_path / "config.yaml", "w") as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
