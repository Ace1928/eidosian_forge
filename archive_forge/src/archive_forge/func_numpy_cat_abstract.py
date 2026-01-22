import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch.testing._internal.autograd_function_db import (
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403
import torch._custom_ops as custom_ops
@custom_ops.impl_abstract('_torch_testing::numpy_cat')
def numpy_cat_abstract(xs, dim):
    assert len(xs) > 0
    assert all((x.device == xs[0].device for x in xs))
    assert all((x.dtype == xs[0].dtype for x in xs))
    return torch.cat(xs, dim=dim)