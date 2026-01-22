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
def sample_inputs_scatter_add(op_info, device, dtype, requires_grad, **kwargs):

    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    yield SampleInput(_tensor((M, S)), 0, _gather((S, S), 1, M), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), 1, _gather((S, S), 0, S), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), -1, _gather((S, S), 0, S), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), 0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))
    yield SampleInput(_tensor((M, S)), 1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))
    yield SampleInput(_tensor((M, S)), -1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))
    yield SampleInput(_tensor(()), 0, zero.clone().detach(), _tensor(()))