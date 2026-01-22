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
def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    yield SampleInput(make_arg((S, S)), make_arg((S,)))
    yield SampleInput(make_arg((S + 1, S)), make_arg((S,)))
    yield SampleInput(make_arg((2, 1, S, S)), make_arg((2, 1, S)))
    yield SampleInput(make_arg((2, 1, S + 1, S)), make_arg((2, 1, S)))
    yield SampleInput(make_arg((0, 0), low=None, high=None), make_arg((0,), low=None, high=None))
    yield SampleInput(make_arg((S, S)), make_arg((0,), low=None, high=None))
    yield SampleInput(make_arg((S, S)), make_arg((S - 2,), low=None, high=None))
    yield SampleInput(make_arg((S, S - 1)), make_arg((S - 2,), low=None, high=None))