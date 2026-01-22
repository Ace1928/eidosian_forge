import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def generate_elementwise_binary_large_value_tensors(op, *, device, dtype, requires_grad=False):
    _large_int_vals = (-1113, 1113, -10701, 10701)
    _large_float16_vals = (-501, 501, -1001.2, 1001.2, -13437.7, 13437.7)
    _large_float_vals = _large_float16_vals + (-4988429.2, 4988429.2, -1e+20, 1e+20)
    l_vals = []
    r_vals = []
    if dtype == torch.float16:
        prod = product(_large_float16_vals, _large_float16_vals)
    elif dtype.is_floating_point:
        prod = product(_large_float_vals, _large_float_vals)
    elif dtype.is_complex:
        complex_vals = product(_large_float_vals, _large_float_vals)
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int16, torch.int32, torch.int64):
        prod = product(_large_int_vals, _large_int_vals)
    else:
        raise ValueError('Unsupported dtype!')
    for l, r in prod:
        l_vals.append(l)
        r_vals.append(r)
    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(lhs, args=(rhs,))