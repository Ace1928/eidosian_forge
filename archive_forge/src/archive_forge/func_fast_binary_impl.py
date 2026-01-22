import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
def fast_binary_impl(mode, *args, **kwargs):

    def slow(msg):
        count_label(f'slow {msg}')
        with mode:
            return slow_ref(*args, **kwargs)
    count_label('attempt fast')
    operands = args
    has_scalars = False
    has_tensors = False
    final_shape = None
    for op in operands:
        shape = op.shape if isinstance(op, torch.Tensor) else ()
        if len(shape) == 0:
            has_scalars = True
        else:
            has_tensors = True
        if final_shape is None:
            final_shape = shape
        final_shape = infer_size(final_shape, shape)
    assert final_shape is not None
    for op in operands:
        if isinstance(op, torch.Tensor) and op.shape == final_shape:
            break
    else:
        return slow('both tensors nontrivially broadcast')
    cpu = torch.device('cpu')
    common_device = cpu
    common_dtype = None
    output_dtype = None
    has_different_input_dtypes = False
    for op in operands:
        if not isinstance(op, torch.Tensor):
            has_different_input_dtypes = True
            continue
        if common_device == cpu and (not op.device.type == 'cpu'):
            common_device = op.device
        if common_dtype is None:
            common_dtype = op.dtype
        elif common_dtype != op.dtype:
            has_different_input_dtypes = True
    if has_different_input_dtypes:
        _, common_dtype = elementwise_dtypes(*operands, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
    current_cpu_scalars_on_non_cpu = 0
    max_cpu_scalars_on_non_cpu = 1
    for op in operands:
        if not isinstance(op, torch.Tensor):
            continue
        if common_device != cpu and op.dim() == 0 and (op.device == cpu):
            if current_cpu_scalars_on_non_cpu >= max_cpu_scalars_on_non_cpu:
                return slow('error')
            current_cpu_scalars_on_non_cpu += 1
        elif op.device != common_device:
            return slow('error')
    is_contiguous = True
    is_channels_last = True
    if is_noncontiguous_supported(common_device):
        for op in operands:
            if not isinstance(op, torch.Tensor):
                continue
            is_contiguous = is_contiguous and op.is_contiguous(memory_format=torch.contiguous_format)
            is_channels_last = is_channels_last and op.is_contiguous(memory_format=torch.channels_last)
    if is_contiguous:
        count_label('fast is_contiguous')
        return FakeTensor(mode, torch.empty(final_shape, dtype=common_dtype, device='meta', memory_format=torch.contiguous_format), device=common_device)
    if is_channels_last:
        count_label('fast channels_last')
        return FakeTensor(mode, torch.empty(final_shape, dtype=common_dtype, device='meta', memory_format=torch.channels_last), device=common_device)
    return slow('no contiguity match')