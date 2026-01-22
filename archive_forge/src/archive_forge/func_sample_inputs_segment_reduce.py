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
def sample_inputs_segment_reduce(op_info, device, dtype, requires_grad, *, mode='lengths', **kwargs):

    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (((S,), 0, [0, 1, 2, 2], False), ((S,), 0, [0, 1, 2, 2], True), ((S,), 0, [2, 0, 3, 0], False), ((S, S), 0, [0, 1, 2, 2], False), ((M, S, S), 0, [1, 2, 0, 6, 0], True), ((S, S), 1, [[0, 1, 2, 2] for _ in range(S)], False), ((S, S), 1, [[2, 0, 3, 0], [0, 1, 2, 2], [3, 0, 2, 0], [1, 1, 1, 2], [0, 1, 2, 2]], False), ((S, S, S), 1, [[0, 1, 2, 2] for _ in range(S)], False), ((S, S, S), 1, [[2, 0, 3, 0], [0, 1, 2, 2], [3, 0, 2, 0], [1, 1, 1, 2], [0, 1, 2, 2]], False))
    reductions = ['max', 'mean', 'min', 'sum', 'prod']
    for args, reduce, initial in product(test_cases, reductions, [1, 2]):
        inp_shape, dim, lengths, unsafe = args
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        sample_input_kwargs = {'axis': dim, 'unsafe': unsafe, 'initial': initial}
        if mode == 'lengths':
            sample_input_kwargs['lengths'] = lengths_t
        elif mode == 'offsets':
            zeros_shape = list(lengths_t.shape)
            zeros_shape[dim] = 1
            offsets_t = torch.cat((lengths_t.new_zeros(zeros_shape), lengths_t), dim).cumsum_(dim)
            sample_input_kwargs['offsets'] = offsets_t
        else:
            raise RuntimeError(f"mode most be one of 'offsets' or 'lengths' got '{mode}'.")
        yield SampleInput(_tensor(inp_shape), args=(reduce,), kwargs=sample_input_kwargs)