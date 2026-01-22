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
def sample_inputs_conv3d(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple = (((1, 1, 4, 4, 4), (1, 1, 1, 1, 1), (1,), {'padding': 'same'}), ((1, 1, 4, 4, 4), (1, 1, 4, 4, 4), (1,), {'stride': (2, 2, 2)}), ((1, 1, 5, 5, 5), (1, 1, 3, 3, 3), (1,), {'dilation': 2}), ((1, 1, 1, 1, 10), (1, 1, 1, 1, 4), None, {'padding': 'valid'}), ((1, 1, 10, 11, 12), (1, 1, 1, 2, 5), None, {'padding': 'same'}), ((1, 1, 10, 11, 12), (1, 1, 1, 2, 5), None, {'padding': 'same', 'dilation': 2}), ((1, 1, 10, 11, 12), (1, 1, 4, 4, 4), None, {'padding': 'same', 'dilation': 3}), ((1, 1, 1, 1, 10), (1, 1, 1, 1, 4), None, {'padding': 'valid'}), ((3, 9, 3, 1, 9), (3, 3, 3, 1, 9), (3,), {'groups': 3}), ((3, 9, 3, 1, 9), (3, 3, 3, 1, 9), (3,), {'stride': (2, 2, 2), 'dilation': 1, 'groups': 3}))
    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)