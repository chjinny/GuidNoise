import torch
from typing import Optional

def differentiable_histogram(input: torch.Tensor, bins: int = 100, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    """
    Reference:
        This implementation is based on below paper and github repository.
        Paper : https://arxiv.org/abs/1611.00822
        Repository: https://github.com/Yukun-Huang/pytorch-differentiable-histogram
        We gratefully acknowledge contribution to making histogram computation differentiable in PyTorch.
    """
    assert input.ndim >= 2
    input = input.view(input.shape[0], input.shape[1], -1)
    batch_size, n_channels, n_values = input.shape

    if min_val is None:
        min_val = input.min().item()
    if max_val is None:
        max_val = input.max().item()
    
    hist = torch.zeros(batch_size, n_channels, bins).to(input.device)

    delta = (max_val - min_val) / bins
    BIN_Table = torch.arange(start=0, end=bins+1, step=1) * delta

    for dim in range(1, bins-1):
        h_curr = BIN_Table[dim].item()
        h_last = BIN_Table[dim - 1].item()
        h_next = BIN_Table[dim + 1].item()

        mask_last = ((h_last <= input) & (input < h_curr)).float()
        mask_next = ((h_curr <= input) & (input <= h_next)).float()

        hist[:, :, dim] += torch.sum(((input - h_last) * mask_last).view(batch_size, n_channels, -1), dim=-1)
        hist[:, :, dim] += torch.sum(((h_next - input) * mask_next).view(batch_size, n_channels, -1), dim=-1)
    
    mask = (input < BIN_Table[1]).float()
    hist[:, :, 0] += torch.sum(((BIN_Table[1] - input) * mask).view(batch_size, n_channels, -1), dim=-1)

    mask = (input >= BIN_Table[bins-1]).float()
    hist[:, :, bins-1] += torch.sum(((input - BIN_Table[bins-1]) * mask).view(batch_size, n_channels, -1), dim=-1)

    hist = hist / delta
    hist = hist / hist.sum(dim=-1, keepdim=True) * n_values
    
    return hist

def refine_loss(pred, target, bins=256, reduction='mean'):
    tp = torch.cat((target, pred), dim=0)
    tp_hist = differentiable_histogram(tp, bins=bins, min_val=0, max_val=1.0)
    tp_hist = tp_hist / tp_hist.sum(-1, keepdim=True)
    t = tp_hist[:len(target)]
    p = tp_hist[len(target):]

    loss = (t*torch.log(t+1e-8) - t*torch.log(p+1e-8)).sum(-1).clip(min=1e-10)
    if loss.mean().isnan():
        raise ValueError("refine_loss is NaN")
    if reduction == 'mean':
        return loss.mean()
    return loss