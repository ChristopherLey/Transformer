import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class Head(nn.Module):
    """
    one head of self-attention
    """

    def __init__(self, input_size: int, head_size: int, block_size: int, dropout: float = 0.5):
        super(Head, self).__init__()
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        # H = head_size
        k = self.key(x)     # (B, T, C) @ (C, H)--> (B, T, H)
        q = self.query(x)   # (B, T, C) @ (C, H)--> (B, T, H)
        # compute attention scores ('affinities')
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = weights @ v   # (B, T, T) @ (B, T, H)
        return out  # (B, T, H)

class MultiHeadAttention(nn.Module):
    """
    Multiple Heads of Self Attention in Parallel
    """
    def __init__(
            self,
            input_size: int,
            block_size: int,
            num_heads: int,
            output_dim: Optional[int] = None,
            dropout: float = 0.5
    ):
        super(MultiHeadAttention, self).__init__()
        head_size = input_size // num_heads
        if output_dim is None:
            output_dim = input_size
        self.heads = nn.ModuleList(
            [Head(input_size, head_size, block_size, dropout=dropout) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(head_size*num_heads, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = torch.cat([head(x) for head in self.heads], dim=-1)     # (B, T, num_heads*head_size)
        x = self.projection(x)
        return self.dropout(x)

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    """
    def __init__(
            self,
            input_size: int,
            hidden_dim: Optional[int] = None,
            output_size: Optional[int] = None,
            dropout: float = 0.5
    ):
        if hidden_dim is None:
            hidden_dim = input_size
        if output_size is None:
            output_size = input_size
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class Block(nn.Module):
    """
    Transformer Block: communication followed by computation
    """
    def __init__(self, input_size: int, block_size: int, num_heads: int, dropout: float = 0.5):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(input_size)   # per token normalisation
        self.self_attention = MultiHeadAttention(input_size, block_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(input_size)
        self.feed_forward = FeedForward(input_size, hidden_dim=input_size*4, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            block_size: int,
            num_heads: int,
            num_blocks: int,
            dropout: float = 0.5    # Can be thought of as training an ensemble of subnets or a consensus mechanism
    ):
        super(TransformerDecoder, self).__init__()
        self.block_size = block_size
        self.token_embeddings_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        block_set: list = [
            Block(input_size=embedding_dim, block_size=block_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]
        block_set.append(nn.LayerNorm(embedding_dim))
        self.blocks = nn.Sequential(
            *block_set
        )
        self.language_modelling_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        token_embedding = self.token_embeddings_table(idx)   # (B, T, embedding_dim)
        positional_embedding = self.position_embedding_table(torch.arange(T, device=idx.device))    # (T, C)
        x = token_embedding + positional_embedding  # broadcasting will copy positional embedding along the batches
        x = self.blocks(x)
        logits = self.language_modelling_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.LongTensor, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size token
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]   # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == '__main__':
    from data.datareader import TinyShakespeare
    reader = TinyShakespeare('input.txt')
    sample = reader[0]
    xb = sample['input'].view(1, -1)
    yb = sample['target'].view(1, -1)
    model = TransformerDecoder(
        reader.vocab_size,
        embedding_dim=32,
        block_size=reader.block_size,
        num_heads=4,
        num_blocks=3
    )
    logits, loss = model(xb, yb)
    print(logits.shape)
    print(loss)

    idx = torch.zeros((1, 1), dtype=torch.long)
    print(reader.decoder(model.generate(idx, max_new_tokens=100)[0].tolist()))
