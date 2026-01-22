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
def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):
    """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
    if self.config.separate_cls:
        cls_pos = pos_id.new_tensor([-2 ** block_index + 1])
        pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
        return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
    else:
        return pos_id[::2]