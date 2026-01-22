from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
def make_coords_grid(batch_size: int, h: int, w: int, device: str='cpu'):
    device = torch.device(device)
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)