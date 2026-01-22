import torch
import functools
from torch.testing import make_tensor
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
def sample_inputs_map(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput([make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2)], args=(make_arg(1, low=0.1, high=2), make_arg(1, low=0.1, high=2)))