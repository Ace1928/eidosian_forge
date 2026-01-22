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
def error_inputs_cross(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    sample = SampleInput(input=make_arg((S, 3)), args=(make_arg((S, 1)),))
    err = 'inputs dimension -1 must have length 3'
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)
    sample = SampleInput(input=make_arg((5, S, 3)), args=(make_arg((S, 3)),))
    err = 'inputs must have the same number of dimensions'
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)
    sample = SampleInput(input=make_arg((S, 2)), args=(make_arg((S, 2)),))
    err = 'must have length 3'
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)
    sample = SampleInput(input=make_arg((S, 2)), args=(make_arg((S, 2)),), kwargs=dict(dim=2))
    err = 'Dimension out of range'
    yield ErrorInput(sample, error_regex=err, error_type=IndexError)