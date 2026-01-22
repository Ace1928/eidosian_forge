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
def sample_inputs_linalg_ldl_factor(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix, random_symmetric_pd_matrix
    device = torch.device(device)
    yield SampleInput(random_symmetric_pd_matrix(S, dtype=dtype, device=device), kwargs=dict(hermitian=False))
    yield SampleInput(random_symmetric_pd_matrix(S, 2, dtype=dtype, device=device), kwargs=dict(hermitian=False))
    yield SampleInput(torch.zeros(0, 0, dtype=dtype, device=device), kwargs=dict(hermitian=False))
    yield SampleInput(torch.zeros(0, 2, 2, dtype=dtype, device=device), kwargs=dict(hermitian=False))
    magma_254_available = device.type == 'cuda' and _get_magma_version() >= (2, 5, 4)
    if dtype.is_complex and (device.type == 'cpu' or magma_254_available):
        yield SampleInput(random_hermitian_pd_matrix(S, dtype=dtype, device=device), kwargs=dict(hermitian=True))
        yield SampleInput(random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device), kwargs=dict(hermitian=True))