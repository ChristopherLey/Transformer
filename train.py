import torch
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from model.transformer_experiment import Transformer_Experiment
from torch.utils.data import DataLoader
from data.datareader import TinyShakespeare


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 256
epochs = 2
lr = 3e-4
data = Path('./data/input.txt')
block_size = 256
num_heads = 8
embedding_dim = 64*num_heads
num_blocks = 6


train_reader = TinyShakespeare(data, version='train', block_size=block_size)
train_dataloader = DataLoader(train_reader, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=12)
val_reader = TinyShakespeare(data, version='val', block_size=block_size)
val_dataloader = DataLoader(val_reader, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=12)

model = Transformer_Experiment(
    vocab_size=train_reader.vocab_size,
    embedding_dim=embedding_dim,
    block_size=train_reader.block_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    dropout=0.5
)

loss_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=4,
        mode="min",
        filename="model-{epoch:02d}-{val_loss:.6f}",
    )

callbacks = [loss_callback, ]

version_path = (
        f"LM-Transformer-{datetime.now().strftime('%d-%m_%H:%M:%S')}"
    )

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir=Path("."),
    version=version_path,
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=[0],
    logger=tb_logger,
    callbacks=callbacks,
    max_epochs=30,
    log_every_n_steps=1
)

trainer.fit(model, train_dataloader, val_dataloader)