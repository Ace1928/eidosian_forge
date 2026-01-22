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
def sample_inputs_matrix_rank(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces inputs for matrix rank that test
    all possible combinations for atol and rtol
    """

    def make_tol_arg(kwarg_type, inp):
        if kwarg_type == 'none':
            return None
        if kwarg_type == 'float':
            return 1.0
        assert kwarg_type == 'tensor'
        return torch.ones(inp.shape[:-2], device=device)
    for tol_type in ['float', 'tensor']:
        for atol_type, rtol_type in product(['none', tol_type], repeat=2):
            if not atol_type and (not rtol_type):
                continue
            for sample in sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad):
                assert sample.kwargs == {}
                sample.kwargs = {'atol': make_tol_arg(atol_type, sample.input), 'rtol': make_tol_arg(rtol_type, sample.input)}
                yield sample
    yield from sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)