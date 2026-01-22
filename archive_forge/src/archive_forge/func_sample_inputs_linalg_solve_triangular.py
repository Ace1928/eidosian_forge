import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def sample_inputs_linalg_solve_triangular(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    bs = (1, 2, 0)
    ns = (3, 0)
    ks = (1, 3, 0)
    for b, n, k, (left, upper, uni) in product(bs, ns, ks, product((True, False), repeat=3)):
        if b == 1:
            A = make_arg((n, n)) if left else make_arg((k, k))
            B = make_arg((n, k))
        else:
            A = make_arg((b, n, n)) if left else make_arg((b, k, k))
            B = make_arg((b, n, k))
        if uni:
            A.diagonal(0, -2, -1).fill_(1.0)
        else:
            d = A.diagonal(0, -2, -1)
            d[d.abs() < 1e-06] = 1.0
        if upper:
            A.triu_()
        else:
            A.tril_()
        kwargs = {'upper': upper, 'left': left, 'unitriangular': uni}
        if requires_grad:
            for grad_A, grad_B in product((True, False), repeat=2):
                if not grad_A and (not grad_B):
                    continue
                yield SampleInput(A.clone().requires_grad_(grad_A), args=(B.clone().requires_grad_(grad_B),), kwargs=kwargs)
        else:
            yield SampleInput(A, args=(B,), kwargs=kwargs)