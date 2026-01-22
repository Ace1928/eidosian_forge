import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@classmethod
def process_kernel(cls, kernel, *args, **kwargs):
    binded_args = signature(kernel).bind(*args, **kwargs).arguments
    args_flat, args_spec = pytree.tree_flatten(binded_args)
    is_arg_tensor = []
    tensor_args = []
    non_tensor_args: List[Any] = []
    for arg in args_flat:
        is_arg_tensor.append(isinstance(arg, IRNode))
        if is_arg_tensor[-1]:
            tensor_args.append(arg)
        else:
            if isinstance(arg, sympy.Expr):
                arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
            non_tensor_args.append(arg)

    def unflatten_args(new_tensor_args, new_non_tensor_args):
        result = []
        it_tensors = iter(new_tensor_args)
        it_non_tensors = iter(new_non_tensor_args)
        for is_tensor in is_arg_tensor:
            if is_tensor:
                result.append(next(it_tensors))
            else:
                result.append(next(it_non_tensors))
        r = pytree.tree_unflatten(result, args_spec)
        return (r.get('args', []), r.get('kwargs', {}))
    tensor_args = [cls.realize_input(x) for x in tensor_args]
    for x in tensor_args:
        if is_storage_and_layout(x):
            as_storage_and_layout(x, freeze=True)
    example_args = []
    for x in tensor_args:
        if x.get_name() in V.graph.constants:
            example_args.append(V.graph.constants[x.get_name()])
        else:
            example_args.append(ir_node_to_tensor(x, guard_shape=True))
    new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
    example_output = kernel(*new_args, **new_kwargs)
    example_out_li = [example_output] if not isinstance(example_output, (list, tuple)) else example_output
    for t in example_out_li:
        if isinstance(t, torch.Tensor) and t.is_sparse:
            V.graph.disable_cudagraphs = True
            msg = 'sparsity not handled. Please file issue for sparse inference weights.'
            if (stack_trace := V.graph.current_node.meta.get('stack_trace', None)):
                msg = f'{msg} Found from : \n {stack_trace}'
            V.graph.disable_cudagraphs_reason = msg
    if maybe_free_unbacked_symbols(example_output):
        example_output = V.graph.current_node.meta['val']
    return (example_output, tensor_args, non_tensor_args, unflatten_args)