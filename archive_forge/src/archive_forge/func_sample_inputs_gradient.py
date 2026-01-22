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
def sample_inputs_gradient(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    test_cases_float = (((S,), None, None, 1), ((S,), 2.0, None, 1), ((S, S), None, None, 2), ((S, S), [2.0, 2.1], None, 1), ((S, S), [2.0, 2.1], (0, 1), 1), ((4, 4, 4), [2.0, 1.0], (0, 1), 2))
    for size, spacing, dim, edge_order in test_cases_float:
        t = make_arg(size)
        yield SampleInput(t, dim=dim, spacing=spacing, edge_order=edge_order)
    test_cases_tensor = (((3, 3, 3), ((1.1, 2.0, 3.5), (4.0, 2, 6.0)), (0, -1), 1), ((3, 3, 3), ((1.0, 3.0, 2.0), (8.0, 6.0, 1.0)), (0, 1), 2))
    for size, coordinates, dim, edge_order in test_cases_tensor:
        t = make_arg(size)
        coordinates_tensor_list = []
        for coords in coordinates:
            a = torch.tensor(coords, device=device)
            coordinates_tensor_list.append(a.to(dtype))
        yield SampleInput(t, dim=dim, spacing=coordinates_tensor_list, edge_order=edge_order)