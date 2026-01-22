import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
@register_decomposition(aten.masked_fill_)
def masked_fill_(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType) -> TensorLikeType:
    b = torch.masked_fill(a, mask, value)
    a.copy_(b)
    return a