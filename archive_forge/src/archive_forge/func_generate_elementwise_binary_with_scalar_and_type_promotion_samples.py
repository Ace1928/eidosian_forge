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
def generate_elementwise_binary_with_scalar_and_type_promotion_samples(op, *, device, dtype, requires_grad=False):
    if op.name in ('eq', 'ne', 'gt', 'ge', 'lt', 'le', 'logical_and', 'logical_or', 'logical_xor'):
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
        shape = (23,)
        values = (float('nan'), float('inf'), -float('inf'))
        scalar_tensors = tuple((torch.tensor(val) for val in values))
        if op.supports_rhs_python_scalar:
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            for scalar in values + scalar_tensors:
                yield SampleInput(lhs, args=(scalar,))
                if op.supports_one_python_scalar:
                    yield SampleInput(scalar, args=(rhs,))