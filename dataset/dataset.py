import torch
from pathlib import Path
import torchvision
import random
def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lq_dir, lq_filter, lq_name_filter, hq_dir, hq_filter, hq_name_filter, crop_size, num_limit, *args, **kwargs):
        self.crop_size = crop_size
        self.lq_dir = Path(lq_dir)
        self.hq_dir = Path(hq_dir)
        self.lq_files = list(self.lq_dir.rglob(lq_filter))
        self.hq_files = list(self.hq_dir.rglob(hq_filter))
        self.lq_files.sort()
        self.hq_files.sort()
        if num_limit is not None:
            random_seed = 42
            random.seed(random_seed)
            random.shuffle(self.lq_files)
            random.seed(random_seed)
            random.shuffle(self.hq_files)
            self.lq_files = self.lq_files[:num_limit]
            self.hq_files = self.hq_files[:num_limit]
        self.lq_files = [file for file in self.lq_files if not file.name.startswith('.')]
        self.hq_files = [file for file in self.hq_files if not file.name.startswith('.')]
        assert len(self.lq_files) == len(self.hq_files)
        for lq, hq in zip(self.lq_files, self.hq_files):
            assert lq.name.replace(lq_name_filter, "") == hq.name.replace(hq_name_filter, "")

    def __len__(self):
        return len(self.lq_files)
    
    def __getitem__(self, idx):
        lq = torchvision.io.read_image(self.lq_files[idx])/255.0
        hq = torchvision.io.read_image(self.hq_files[idx])/255.0
        assert lq.shape == hq.shape, (f'Image shapes are different: {lq.shape}, {hq.shape}.')
        h, w = lq.shape[-2:]
        i = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        j = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        lq = lq[..., i:i+self.crop_size, j:j+self.crop_size]
        hq = hq[..., i:i+self.crop_size, j:j+self.crop_size]
        return {
            "lq": lq,
            "hq": hq,
            "lq_path": str(self.lq_files[idx]),
            "hq_path": str(self.hq_files[idx]),
        }