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
@custom_ops.impl_backward('_torch_testing::numpy_view_copy')
def numpy_view_copy_backward(ctx, x_shape, grad_out) -> Tensor:
    return {'x': torch.ops._torch_testing.numpy_view_copy(grad_out, x_shape)}