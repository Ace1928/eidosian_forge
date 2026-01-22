import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def weight_bias_name(ihhh, layer, suffix):
    weight_name = f'weight_{ihhh}_l{layer}{suffix}'
    bias_name = f'bias_{ihhh}_l{layer}{suffix}'
    return (weight_name, bias_name)