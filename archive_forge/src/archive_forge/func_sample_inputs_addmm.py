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
def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2)
    tests_list = [((2, 3), (2, 2), (2, 3), False)]
    tests_with_lhs_broadcasting = [((1,), (2, 2), (2, 3), True), ((), (2, 2), (2, 3), True)]
    test_cases = tests_list + tests_with_lhs_broadcasting
    kwargs = dict(alpha=alpha_val, beta=beta_val)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape_a, shape_b, shape_c, broadcasts_input in test_cases:
        yield SampleInput(make_arg(shape_a), make_arg(shape_b), make_arg(shape_c), **kwargs).with_metadata(broadcasts_input=broadcasts_input)
    if dtype.is_complex:
        shape = (3, 3)
        yield SampleInput(make_arg(shape), make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad), make_arg(shape), **kwargs)
        yield SampleInput(make_arg(shape), make_arg(shape), make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad), **kwargs)