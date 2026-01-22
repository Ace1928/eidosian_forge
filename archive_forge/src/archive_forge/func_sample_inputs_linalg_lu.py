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
def sample_inputs_linalg_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    full_rank = op_info.name == 'linalg.lu_factor'
    make_fn = make_tensor if not full_rank else make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    def out_fn(output):
        if op_info.name == 'linalg.lu':
            return (output[1], output[2])
        else:
            return output
    batch_shapes = ((), (3,), (3, 3))
    pivots = (True, False) if torch.device(device).type == 'cuda' else (True,)
    deltas = (-2, -1, 0, +1, +2)
    for batch_shape, pivot, delta in product(batch_shapes, pivots, deltas):
        shape = batch_shape + (S + delta, S)
        A = make_arg(shape) if not full_rank else make_arg(*shape)
        yield SampleInput(A, kwargs={'pivot': pivot}, output_process_fn_grad=out_fn)