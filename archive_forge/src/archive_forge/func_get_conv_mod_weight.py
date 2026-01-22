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
def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return mod.weight.detach()
    elif isinstance(mod, (nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d)):
        return mod[0].weight.detach()
    else:
        return mod._weight_bias()[0]