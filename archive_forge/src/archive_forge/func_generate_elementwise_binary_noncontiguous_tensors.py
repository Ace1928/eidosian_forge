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
def generate_elementwise_binary_noncontiguous_tensors(op, *, device, dtype, requires_grad=False, exclude_zero=False):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    lhs = make_arg((1026,), noncontiguous=True, **op.lhs_make_tensor_kwargs)
    rhs = make_arg((1026,), noncontiguous=True, **op.rhs_make_tensor_kwargs)
    yield SampleInput(lhs.clone(), args=(rhs.clone(),))
    yield SampleInput(lhs.contiguous(), args=(rhs,))
    lhs = make_arg((789, 357), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((789, 357), **op.rhs_make_tensor_kwargs)
    yield SampleInput(lhs.T, args=(rhs.T,))
    shapes = ((5, 7), (1024,))
    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
        lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        lhs_non_contig.copy_(lhs)
        rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        rhs_non_contig.copy_(rhs)
        yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
        yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))
    shape = (2, 2, 1, 2)
    lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
    rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
    lhs_non_contig = lhs[:, 1, ...]
    rhs_non_contig = rhs[:, 1, ...]
    yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
    yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))
    shapes = ((1, 3), (1, 7), (5, 7))
    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
        lhs_non_contig = lhs.expand(3, -1, -1)
        rhs_non_contig = rhs.expand(3, -1, -1)
        yield SampleInput(lhs_non_contig, args=(rhs_non_contig,))