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
def sample_inputs_einsum(op_info, device, dtype, requires_grad=False, **kwargs):

    def c(t):
        return t.clone().requires_grad_(requires_grad)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    x = make_arg((3,))
    y = make_arg((4,))
    A = make_arg((2, 3))
    B = make_arg((1, 3))
    C = make_arg((1, 2, 3))
    D = make_arg((1, 3, 4))
    E = make_arg((4, 4))
    H = make_arg((3, 3))
    I = make_arg((1, 3, 1))
    yield SampleInput([c(x)], 'i->')
    yield SampleInput([c(x), c(y)], 'i,j->ij')
    yield SampleInput([c(A)], 'ij->i')
    yield SampleInput([c(A), c(B)], 'ij,kj->ik')
    yield SampleInput([c(A), c(E)], 'ij,Ab->ijAb')
    yield SampleInput([c(C), c(D)], 'aij,ajk->aik')
    yield SampleInput([c(D), c(E)], 'aij,jk->aik')
    yield SampleInput([c(C), c(B)], 'ijk,ik->j')
    yield SampleInput([c(I)], 'iji->j')
    yield SampleInput([c(H)], 'i...->...')
    yield SampleInput([c(C), c(x)], '...ik, ...j -> ij')