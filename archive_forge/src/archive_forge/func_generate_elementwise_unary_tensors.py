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
def generate_elementwise_unary_tensors(op, *, device, dtype, requires_grad, **kwargs):
    if dtype is torch.bool:
        tensors = (torch.empty(0, device=device, dtype=torch.bool), torch.tensor(True, device=device), torch.tensor(False, device=device), torch.tensor((True, False), device=device), make_tensor((812,), device=device, dtype=dtype), make_tensor((1029, 917), device=device, dtype=dtype))
        for a in tensors:
            yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])
    shapes = ((1029, 917), (812,), (0,), (0, 3, 3), (1, 0, 5), (6, 0, 0, 0), (3, 0, 1, 0))
    make_arg = partial(_make_unary_elementwise_tensor, op=op, device=device, dtype=dtype, requires_grad=requires_grad)
    for shape in shapes:
        a = make_arg(shape)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])