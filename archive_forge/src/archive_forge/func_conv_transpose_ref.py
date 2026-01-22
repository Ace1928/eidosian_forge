from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def conv_transpose_ref(input, weight, bias, stride=1, padding=0, output_padding=0, dilation=1, groups=1, fn=None):
    assert fn is not None
    grad_fn_map = {torch.nn.functional.conv_transpose1d: torch.nn.grad.conv1d_input, torch.nn.functional.conv_transpose2d: torch.nn.grad.conv2d_input, torch.nn.functional.conv_transpose3d: torch.nn.grad.conv3d_input}
    batched_dim_map = {torch.nn.functional.conv_transpose1d: 3, torch.nn.functional.conv_transpose2d: 4, torch.nn.functional.conv_transpose3d: 5}
    input, weight = (torch.from_numpy(input), torch.from_numpy(weight))
    is_batched = len(input.shape) == batched_dim_map[fn]
    if not is_batched:
        input = input.unsqueeze(0)
    if bias is not None:
        bias = torch.from_numpy(bias)
        unsqueeze_dims = input.ndim - 2
        for _ in range(unsqueeze_dims):
            bias = bias.unsqueeze(1)
    grad_output = input
    conv_transpose_output = fn(grad_output.to('meta'), weight.to('meta'), None, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    input_size = conv_transpose_output.shape
    grad_fn = grad_fn_map[fn]
    if weight.dtype.is_complex:
        out = complex_conv(grad_fn, input_size, weight, grad_output, stride, padding, dilation, groups)
    else:
        out = grad_fn(input_size, weight, grad_output, stride, padding, dilation, groups)
    if bias is not None:
        out = out + bias
    return out.squeeze(0) if not is_batched else out