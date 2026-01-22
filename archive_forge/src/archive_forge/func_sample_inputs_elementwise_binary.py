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
def sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs):
    _M = S if kwargs.get('small_inputs_only', False) else M
    _S = XS if kwargs.get('small_inputs_only', False) else S
    if hasattr(op, 'rhs_make_tensor_kwargs'):
        exclude_zero = op.rhs_make_tensor_kwargs.get('exclude_zero', False)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=exclude_zero)
    shapes = (((), ()), ((_S,), ()), ((_S, 1), (_S,)), ((_M, _S), ()), ((_S, _M, _S), (_M, _S)), ((_S, _M, _S), (_S, _M, _S)), ((_M, 1, _S), (_M, _S)), ((_M, 1, _S), (1, _M, _S)), ((0, 1, XS), (0, _M, XS)))
    sample_kwargs = kwargs.get('sample_kwargs', {})
    for shape_lhs, shape_rhs in shapes:
        lhs = make_arg(shape_lhs, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape_rhs, **op.rhs_make_tensor_kwargs)
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)
        yield SampleInput(lhs, args=(rhs,), kwargs=sample_kwargs, broadcasts_input=broadcasts_input)