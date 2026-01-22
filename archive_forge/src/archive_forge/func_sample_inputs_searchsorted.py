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
def sample_inputs_searchsorted(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = (((0,), ((0,),), False), ((M,), ((), (M,), (M, M)), False), ((0, 0), ((0, 0),), False), ((M, M), ((M, M),), False), ((0, 0, 0), ((0, 0, 0),), False), ((M, M, M), ((M, M, M),), False), ((L,), ((),), True))
    for (size, input_sizes, is_scalar), noncontiguous, out_int32, right in product(sizes, [False, True], [False, True], [False, True]):
        unsorted_tensor = make_arg(size, noncontiguous=noncontiguous)
        for input_size in input_sizes:
            input = make_arg(input_size, noncontiguous=noncontiguous)
            if is_scalar:
                input = input.item()
            if np.prod(size) == 0:
                boundary_tensor = unsorted_tensor
                sorter = make_tensor(size, dtype=torch.int64, device=device, noncontiguous=noncontiguous)
            else:
                boundary_tensor, sorter = torch.sort(unsorted_tensor)
            side = 'right' if right else 'left'
            yield SampleInput(boundary_tensor, input, out_int32=out_int32, right=right)
            yield SampleInput(boundary_tensor, input, out_int32=out_int32, side=side)
            yield SampleInput(unsorted_tensor, input, out_int32=out_int32, right=right, sorter=sorter)
            yield SampleInput(unsorted_tensor, input, out_int32=out_int32, side=side, sorter=sorter)