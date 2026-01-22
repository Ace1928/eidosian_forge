import unittest
from functools import partial
from itertools import product
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import all_types_and, floating_types
from torch.testing._internal.common_utils import TEST_SCIPY, torch_to_numpy_dtype_dict
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
from torch.testing._internal.opinfo.utils import (
def sample_inputs_i0_i1(op_info, device, dtype, requires_grad, **kwargs):
    exclude_zero = requires_grad and op_info.op == torch.special.i0e
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, exclude_zero=exclude_zero)
    yield SampleInput(make_arg((S,)))
    yield SampleInput(make_arg(()))
    if requires_grad and (not exclude_zero):
        t = make_arg((S,))
        t[0] = 0
        yield SampleInput(t)