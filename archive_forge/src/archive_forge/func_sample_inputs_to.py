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
def sample_inputs_to(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    devices = [device]
    if torch.device(device).type == 'cpu':
        devices = [torch.device('cpu'), torch.device('cuda:0')] if torch.cuda.is_available() else devices
    memory_formats = [torch.preserve_format, torch.channels_last]
    for device, nb, cp, mem_f in product(devices, [True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        yield SampleInput(make_arg((S, S, S, S)), args=(device, torch.float64, nb, cp), kwargs=kwargs)
    for nb, cp, mem_f in product([True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        yield SampleInput(make_arg((S, S, S, S)), args=(torch.float64, nb, cp), kwargs=kwargs)
    for device, nb, cp, mem_f in product(devices, [True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        other = make_arg((S, S, S, S), dtype=torch.float64, device=device)
        yield SampleInput(make_arg((S, S, S, S)), args=(other, nb, cp), kwargs=kwargs)