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
@register_decomposition(aten.lerp)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('start', 'end', 'weight'), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def lerp(start: Tensor, end: Tensor, weight: Union[Tensor, NumberType]):
    inputs = [start, end]
    if isinstance(weight, Number):
        weight = start.new_full((), weight)
    else:
        inputs.append(weight)
    assert isinstance(weight, Tensor)
    mask = weight.abs() >= 0.5
    coeff = torch.where(mask, weight - 1, weight)
    base = torch.where(mask, end, start)
    output = coeff * (end - start) + base
    stride = utils.compute_elementwise_output_strides(*_maybe_broadcast(*inputs))
    if output.stride() != stride:
        output = prims.copy_strided(output, stride)
    return handle_noncontiguous_outputs(inputs, output)