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
def sample_inputs_matmul(op_info, device, dtype, requires_grad, is_rmatmul=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    test_cases = (((L,), (L,)), ((S, M), (M,)), ((M,), (M, S)), ((S, M), (M, S)), ((S, 0), (0, M)), ((S, S, M), (M,)), ((S, S, M), (M, S)), ((S, S, 0), (0, S)), ((M,), (S, M, S)), ((S, M), (S, M, S)), ((0, 0), (S, 0, 0)), ((S, S, M, M), (S, S, M, S)), ((S, S, M, M), (M,)), ((M,), (S, S, M, S)), ((S, S, S), (1, S, S)))
    for lhs_shape, rhs_shape in test_cases:
        lhs = make_arg(lhs_shape)
        rhs = make_arg(rhs_shape)
        if not is_rmatmul:
            yield SampleInput(lhs, rhs)
        else:
            yield SampleInput(rhs, lhs)