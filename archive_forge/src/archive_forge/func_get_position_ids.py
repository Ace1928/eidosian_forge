from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def get_position_ids(self, x: Tensor) -> Tensor:
    if self.model_type == 'roberta':
        mask = x.ne(self.padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
    return self.position_ids[:, :x.shape[1]]