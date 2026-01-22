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
@register_op_impl(lambda func: func is torch.ops.aten._local_scalar_dense.default)
def local_scalar_dense(fake_mode, func, arg):
    if fake_mode.shape_env is None or not fake_mode.shape_env.allow_scalar_outputs:
        raise DataDependentOutputException(func)
    if is_float_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symfloat()
    elif is_integer_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symint()
    elif is_boolean_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symbool()
    else:
        raise NotImplementedError(f'local_scalar_dense/item NYI for {arg.dtype}')