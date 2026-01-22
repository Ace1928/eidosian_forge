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
def sample_inputs_linalg_vecdot(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    batches = ((), (0,), (1,), (5,))
    ns = (0, 1, 3, 5)
    for b, n in product(batches, ns):
        shape = b + (n,)
        yield SampleInput(make_arg(shape), args=(make_arg(shape),))
        for i in range(len(shape)):
            yield SampleInput(make_arg(shape), args=(make_arg(shape),), kwargs=dict(dim=i))