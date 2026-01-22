import unittest
from collections.abc import Sequence
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy
def sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, with_dtype=False, use_zero_dimensions=True, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((S,), (0,)), ((S, S), (0,)), ((S, S), (1,)), ((S, S), (-1,)), ((S, M, S), (2,)), *([((S, 0, 0), (-1,))] if use_zero_dimensions else [])]
    kwargs = dict(dtype=torch.float64) if with_dtype else None
    if torch.device(device).type != 'xla':
        cases.append(((), (0,)))
    return (SampleInput(make_arg(shape), args=dim, kwargs=kwargs) for shape, dim in cases)