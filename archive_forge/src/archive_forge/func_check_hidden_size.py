import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int], msg: str='Expected hidden size {}, got {}') -> None:
    if hx.size() != expected_hidden_size:
        raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))