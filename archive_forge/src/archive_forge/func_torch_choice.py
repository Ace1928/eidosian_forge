import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
def torch_choice(self, k: List[int]) -> int:
    """
        Implements `random.choice` via torch ops, so it can be compiled with
        TorchScript and we use PyTorch's RNG (not native RNG)
        """
    index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
    return k[index]