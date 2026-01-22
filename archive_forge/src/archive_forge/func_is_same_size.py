from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.is_same_size])
def is_same_size(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f'__torch_dispatch__, {func}', len_args=2)
    return _get_data(args[0]).is_same_size(_get_data(args[1]))