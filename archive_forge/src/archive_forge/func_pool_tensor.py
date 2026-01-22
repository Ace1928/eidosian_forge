import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
def pool_tensor(self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], mode: str='mean', stride: int=2) -> torch.Tensor:
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None
    if isinstance(tensor, (tuple, list)):
        return type(tensor)((self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor))
    if self.config.separate_cls:
        suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
        tensor = torch.cat([tensor[:, :1], suffix], dim=1)
    ndim = tensor.ndim
    if ndim == 2:
        tensor = tensor[:, None, :, None]
    elif ndim == 3:
        tensor = tensor[:, None, :, :]
    stride = (stride, 1)
    if mode == 'mean':
        tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == 'max':
        tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == 'min':
        tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
    else:
        raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")
    if ndim == 2:
        return tensor[:, 0, :, 0]
    elif ndim == 3:
        return tensor[:, 0]
    return tensor