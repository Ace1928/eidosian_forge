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
def register_op_impl(run_impl_check: Union[Callable[[OpOverload], bool], OpOverload]):

    def impl_decorator(op_impl):
        global op_implementations
        if isinstance(run_impl_check, OpOverload):
            op_implementations.append((lambda func: func == run_impl_check, op_impl))
        else:
            op_implementations.append((run_impl_check, op_impl))
        return op_impl
    return impl_decorator