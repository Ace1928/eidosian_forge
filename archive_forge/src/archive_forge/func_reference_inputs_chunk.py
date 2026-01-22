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
def reference_inputs_chunk(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_chunk(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((13, 9, 11), 17, -1), ((13, 9, 11), 11, -1), ((13,), 12, -1), ((15,), 12, -1), ((15,), 7, 0), ((15,), 9, 0), ((3, 7), 9, 1), ((3, 7), 9, 0), ((3, 7), 2, 0), ((3, 7), 3, 0), ((3, 7), 1, 0), ((3, 7), 1, 1), ((4, 4), 2, 0))
    for shape, chunks, dim in cases:
        yield SampleInput(make_arg(shape), args=(chunks, dim))