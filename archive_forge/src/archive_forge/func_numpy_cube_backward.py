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
@custom_ops.impl_backward('_torch_testing::numpy_cube')
def numpy_cube_backward(ctx, saved, grad_out, grad_dx):
    x, dx = saved
    grad_x = torch.ops._torch_testing.numpy_mul(grad_out, dx) + 6 * torch.ops._torch_testing.numpy_mul(grad_dx, x)
    return {'x': grad_x}