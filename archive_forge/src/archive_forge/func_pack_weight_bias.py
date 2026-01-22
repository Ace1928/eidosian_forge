import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def pack_weight_bias(qweight, bias, dtype):
    if dtype == torch.qint8:
        packed_weight = torch.ops.quantized.linear_prepack(qweight, bias)
        return packed_weight
    else:
        packed_weight = torch.ops.quantized.linear_prepack_fp16(qweight, bias)
        return packed_weight