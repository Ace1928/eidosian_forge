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
def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)
    is_linalg_svd = 'linalg.svd' in op_info.name
    batches = [(), (0,), (3,)]
    ns = [0, 3, 5]

    def uniformize(usv):
        S = usv[1]
        k = S.shape[-1]
        U = usv[0][..., :k]
        Vh = usv[2] if is_linalg_svd else usv[2].mH
        Vh = Vh[..., :k, :]
        return (U, S, Vh)

    def fn_U(usv):
        U, _, _ = uniformize(usv)
        return U.abs()

    def fn_S(usv):
        return uniformize(usv)[1]

    def fn_Vh(usv):
        _, S, Vh = uniformize(usv)
        return (S, Vh.abs())

    def fn_UVh(usv):
        U, S, Vh = uniformize(usv)
        return (U @ Vh, S)
    fns = (fn_U, fn_S, fn_Vh, fn_UVh)
    fullmat = 'full_matrices' if is_linalg_svd else 'some'
    for batch, n, k, fullmat_val, fn in product(batches, ns, ns, (True, False), fns):
        shape = batch + (n, k)
        yield SampleInput(make_arg(*shape), kwargs={fullmat: fullmat_val}, output_process_fn_grad=fn)