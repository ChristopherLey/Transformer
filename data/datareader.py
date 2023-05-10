import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union


class TinyShakespeare(Dataset):
    valid_versions = ['train', 'val']
    target_file = 'input.txt'

    def __init__(
            self,
            path: Union[Path, str],
            block_size: int = 8,
            version: str = 'train',
            split: float = 0.9
    ):
        path = Path(path)
        assert path.is_file(), f"No file named {self.target_file} in {path}"
        assert version in self.valid_versions, f'Only version = {" or ".join(self.valid_versions)} is valid'
        assert 0.0 <= split <= 1.0, f'split must be in the range (0, 1]'
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.raw_data = torch.tensor(self.encoder(text), dtype=torch.long)
        n = int(split*len(self.raw_data))
        if version == 'train':
            self.data = self.raw_data[:n]
        else:
            self.data = self.raw_data[n:]
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size - 1

    def encoder(self, string: str):
        encoded = []
        for character in string:
            encoded.append(self.char_to_idx[character])
        return encoded

    def decoder(self, index_list: list):
        decoded = []
        for index in index_list:
            decoded.append(self.idx_to_char[index])
        return ''.join(decoded)

    def __getitem__(self, idx):
        return {
            'input': self.data[idx:idx + self.block_size],
            'target': self.data[idx+1: idx + self.block_size+1]
        }


if __name__ == "__main__":
    reader = TinyShakespeare(Path('input.txt'))
    x = reader.data[:reader.block_size]
    y = reader.data[1:reader.block_size + 1]
    for t in range(reader.block_size):
        context = x[:t+1]
        target = y[t]
        print(f'when input is {context} the target in {target}')
