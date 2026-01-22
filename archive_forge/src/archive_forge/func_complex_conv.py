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
def complex_conv(fn, input_size, weight, grad_output, stride, padding, dilation, groups):
    grad_output_ = torch.view_as_real(grad_output)
    grad_output_r = grad_output_[..., 0]
    grad_output_i = grad_output_[..., 1]
    weight_ = torch.view_as_real(weight)
    weight_r = weight_[..., 0]
    weight_i = weight_[..., 1]
    a = fn(input_size, weight_r, grad_output_r, stride, padding, dilation, groups)
    b = fn(input_size, weight_i, grad_output_i, stride, padding, dilation, groups)
    c = fn(input_size, weight_r + weight_i, grad_output_r + grad_output_i, stride, padding, dilation, groups)
    return a - b + 1j * (c - a - b)