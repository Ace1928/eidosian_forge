import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
from torch.fx import GraphModule
from torch.fx.graph import Node
from .utils import (
from .ns_types import (
from typing import List, Optional, Dict, Callable
def get_lstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for idx, param_name in enumerate(mod._flat_weights_names):
        if 'weight_ih_l' in param_name or 'weight_hh_l' in param_name:
            param_value = mod._flat_weights[idx].detach()
            res.append(param_value)
    return res