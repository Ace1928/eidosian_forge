from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_unique(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    for shape, sorted, return_inverse, return_counts, dim in product(sizes, [False, True], [False, True], [False, True], [None, -2, -1, 0, 1, 2]):
        if 0 in shape and shape.index(0) is not dim:
            continue
        if dim is not None and (dim < -len(shape) or dim >= len(shape)):
            continue
        kwargs = dict(sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
        input_t = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        yield SampleInput(input_t, **kwargs)
        input_t = make_arg(shape, dtype=torch.bool, requires_grad=False).to(dtype).requires_grad_(requires_grad)
        yield SampleInput(input_t, **kwargs)
        yield SampleInput(make_arg(shape), **kwargs)