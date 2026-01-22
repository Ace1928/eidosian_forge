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
def stride_pool(self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], axis: Union[int, Tuple[int], List[int]]) -> torch.Tensor:
    """
        Perform pooling by stride slicing the tensor along the given axis.
        """
    if tensor is None:
        return None
    if isinstance(axis, (list, tuple)):
        for ax in axis:
            tensor = self.stride_pool(tensor, ax)
        return tensor
    if isinstance(tensor, (tuple, list)):
        return type(tensor)((self.stride_pool(x, axis) for x in tensor))
    axis %= tensor.ndim
    axis_slice = slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
    enc_slice = [slice(None)] * axis + [axis_slice]
    if self.config.separate_cls:
        cls_slice = [slice(None)] * axis + [slice(None, 1)]
        tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)
    return tensor[enc_slice]