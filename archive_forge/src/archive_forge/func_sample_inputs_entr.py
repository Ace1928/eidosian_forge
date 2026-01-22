import unittest
from functools import partial
from itertools import product
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import all_types_and, floating_types
from torch.testing._internal.common_utils import TEST_SCIPY, torch_to_numpy_dtype_dict
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
from torch.testing._internal.opinfo.utils import (
def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    low, _ = op_info.domain
    if requires_grad:
        low = 0 + op_info._domain_eps
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=low, requires_grad=requires_grad)
    yield SampleInput(make_arg((L,)))
    yield SampleInput(make_arg(()))