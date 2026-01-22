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
def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)
    for shape, n in product(tensor_shapes, ns):
        yield SampleInput(make_arg(shape), args=(n,))