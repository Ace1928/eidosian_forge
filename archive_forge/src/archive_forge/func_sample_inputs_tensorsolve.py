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
def sample_inputs_tensorsolve(op_info, device, dtype, requires_grad, **kwargs):
    a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
    dimss = [None, (0, 2)]
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for a_shape, dims in itertools.product(a_shapes, dimss):
        a = make_arg(a_shape)
        b = make_arg(a_shape[:2])
        yield SampleInput(a, b, dims=dims)