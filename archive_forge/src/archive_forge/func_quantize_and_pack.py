import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def quantize_and_pack(w, b):
    weight_observer = weight_observer_method()
    weight_observer(w)
    qweight = _quantize_weight(w.float(), weight_observer)
    packed_weight = torch.ops.quantized.linear_prepack(qweight, b)
    return packed_weight