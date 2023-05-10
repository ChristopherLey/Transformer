import pytorch_lightning as pl
import torch
from torch.optim import Adam
from model.transformer import TransformerDecoder
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Transformer_Experiment(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            block_size: int,
            num_heads: int,
            num_blocks: int,
            dropout: float = 0.5  # Can be thought of as training an ensemble of subnets or a consensus mechanism
    ):
        super().__init__()
        self.model = TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            block_size=block_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout
        )

    def get_word_list(self, index: torch.LongTensor):
        word_list = []
        for i in range(index.shape[0]):
            word_list.append(self.ix_to_word[index[i].item()])
        return word_list

    def training_step(self, sample: dict, sample_idx: int):
        logits, loss = self.model(sample['input'].to(self.device), sample['target'].to(self.device))
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, sample: dict, sample_idx: int):
        logits, loss = self.model(sample['input'].to(self.device), sample['target'].to(self.device))
        self.log("val_loss", loss.item(), prog_bar=True)

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=0.0001)
        return optimiser
