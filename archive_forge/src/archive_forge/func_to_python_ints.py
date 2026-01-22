import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
def to_python_ints(argnums):
    if not isinstance(argnums, (ConstantVariable, TupleVariable)):
        raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
    if isinstance(argnums, ConstantVariable):
        if not isinstance(argnums.value, (int, tuple)):
            raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
        return argnums.value
    else:
        const_vars = argnums.unpack_var_sequence(tx)
        if not all((isinstance(var, ConstantVariable) and isinstance(var.value, int) for var in const_vars)):
            raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to contain int only. Got {const_vars}.')
        return tuple((var.value for var in const_vars))