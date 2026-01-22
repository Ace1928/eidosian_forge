from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.is_strides_like_format])
def is_strides_like_format(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError('MaskedTensors with sparse data do not have is_strides_like_format')
    return func(data, *args[1:], **kwargs)