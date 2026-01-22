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
def sample_inputs_bucketize(op_info, device, dtype, requires_grad, reference_inputs_mode=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = (((), S), ((S,), S), ((S, S), S), ((S, S, S), S), ((S, 1, S), S), ((S, 0, S), S))
    if reference_inputs_mode:
        sizes += (((256,), 128), ((128,), 256), ((32, 32), 11), ((32, 4, 32), 33))
    for (input_shape, nb), out_int32, right in product(sizes, [False, True], [False, True]):
        input_tensor = make_arg(input_shape)
        boundaries = make_arg(nb).msort()
        yield SampleInput(input_tensor, boundaries, out_int32=out_int32, right=right)