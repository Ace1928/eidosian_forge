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
def sample_inputs_linalg_norm(op_info, device, dtype, requires_grad, *, variant=None, **kwargs):
    if variant is not None and variant not in ('subgradient_at_zero',):
        raise ValueError(f"Unsupported variant, expected variant to be 'subgradient_at_zero' but got: {variant}")
    test_sizes = [(S,), (0,), (S, S), (0, 0), (S, 0), (0, S), (S, S, S), (0, S, S), (S, 0, S), (0, 0, 0)]
    vector_ords = (None, 0, 0.5, 1, 2, 3.5, inf, -0.5, -1, -2, -3.5, -inf)
    if dtype in {torch.float16, torch.bfloat16, torch.complex32}:
        matrix_ords = ('fro', inf, -inf, 1, -1)
    else:
        matrix_ords = (None, 'fro', 'nuc', inf, -inf, 1, -1, 2, -2)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, low=None, high=None)
    for test_size in test_sizes:
        is_vector_norm = len(test_size) == 1
        is_matrix_norm = len(test_size) == 2
        is_valid_for_p2 = is_vector_norm or (test_size[-1] != 0 and test_size[-2] != 0)
        for keepdim in [False, True]:
            if variant != 'subgradient_at_zero' and is_valid_for_p2:
                yield SampleInput(make_arg(test_size), keepdim=keepdim)
            if not (is_vector_norm or is_matrix_norm):
                continue
            ords = vector_ords if is_vector_norm else matrix_ords
            for ord in ords:
                if is_vector_norm and test_size[-1] == 0:
                    if ord == np.inf or (ord is not None and ord < 0):
                        continue
                elif is_matrix_norm:
                    dims_to_check = {None: (0,), np.inf: (0,), 2: (0, 1), 1: (1,), -1: (1,), -2: (0, 1), -np.inf: (0,)}.get(ord, ())
                    if any((test_size[d] == 0 for d in dims_to_check)):
                        continue
                if variant == 'subgradient_at_zero':
                    yield SampleInput(torch.zeros(test_size, dtype=dtype, device=device, requires_grad=requires_grad), ord, keepdim=keepdim)
                else:
                    yield SampleInput(make_arg(test_size), ord, keepdim=keepdim)
                    if ord in ['nuc', 'fro']:
                        yield SampleInput(make_arg(test_size), ord=ord, keepdim=keepdim, dim=(0, 1))