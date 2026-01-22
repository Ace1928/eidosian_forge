import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    from .variables import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')
    torch_object_ids = dict()
    ctx_mamager_classes = set()
    c_binding_in_graph_functions = set()
    non_c_binding_in_graph_functions = set()
    torch_name_rule_map = dict()

    def heuristic_record_if_ctx_manager(obj, module, name):
        if issubclass(type(obj), type) and hasattr(obj, '__enter__') and hasattr(obj, '__exit__'):
            torch_name_rule_map[f'{module.__name__}.{name}'] = TorchCtxManagerClassVariable
            ctx_mamager_classes.add(obj)

    def is_special_functions(obj):
        return hashable(obj) and obj in {torch._C._cuda_isCurrentStreamCapturing, torch._C._graph_pool_handle}

    def heuristic_record_if_in_graph_function(obj, module, name):
        try:
            if hasattr(obj, '__wrapped__'):
                obj = obj.__wrapped__
        except Exception:
            pass
        if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)) or is_special_functions(obj):
            torch_name_rule_map[f'{module.__name__}.{name}'] = TorchInGraphFunctionVariable
            if c_binding_only:
                if not hasattr(obj, '__code__'):
                    c_binding_in_graph_functions.add(obj)
            elif hasattr(obj, '__code__'):
                non_c_binding_in_graph_functions.add(obj)
            else:
                c_binding_in_graph_functions.add(obj)

    def _is_allowed_module_prefix(obj):
        allowed_modules = ('torch', 'math')
        disallowed_modules = ['torch.optim', 'torch.nn.modules.rnn', 'torch._dynamo', 'torch._C._dynamo', 'torch._inductor', 'torch._C.inductor', 'torch.fx', 'torch._C._autograd', 'torch._C._cudart', 'torch._C._distributed_autograd', 'torch._C._distributed_c10d', 'torch._C._distributed_rpc', 'torch._C._functorch', 'torch._C._monitor', 'torch._C._nvtx', 'torch._C._lazy', 'torch._C._profiler', 'torch.__config__', 'torch._custom_op', 'torch._dispatch', 'torch._export', 'torch._functorch.make_functional', 'torch._functorch.compile_utils', 'torch._functorch.partitioners', 'torch._functorch.aot_autograd', 'torch._functorch.compilers', 'torch._functorch.fx_minifier', 'torch.autograd.profiler_util', 'torch.autograd.profiler', 'torch._jit_internal', 'torch._library', 'torch._lobpcg', 'torch._logging', 'torch._meta_registrations', 'torch._namedtensor_internals', 'torch._numpy', 'torch._sources', 'torch._subclasses', 'torch._tensor', 'torch._tensor_str', 'torch._utils', 'torch._utils_internal', 'torch._vmap_internals', 'torch.compiler', 'torch.distributed', 'torch.export', 'torch.hub', 'torch.jit', 'torch.library', 'torch.masked.maskedtensor', 'torch.nn.init', 'torch.nn.modules.module', 'torch.nn.parallel', 'torch.nn.utils', 'torch.multiprocessing', 'torch.onnx', 'torch.overrides', 'torch.package', 'torch.profiler', 'torch.serialization', 'torch.storage', 'torch.utils']
        if config.trace_distributed:
            disallowed_modules.append('torch.distributed.')
        allowed_modules_dot = tuple([x + '.' for x in allowed_modules])
        module = inspect.getmodule(obj)
        if module is None:
            return False
        mod_name = module.__name__
        if any((mod_name.startswith(m) for m in disallowed_modules)):
            return False
        return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)

    def _find_torch_objects(module):
        if any((module.__name__.startswith(mod_name) for mod_name in config.allowed_functions_module_string_ignorelist)):
            return
        torch_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                import torch._ops
                if isinstance(obj, torch._ops.HigherOrderOperator):
                    continue
                if obj in (torch.func.grad, deprecated_func.grad, torch.func.vmap, deprecated_func.vmap, torch.nn.functional.triplet_margin_with_distance_loss, torch.cond):
                    continue
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith('torch.') and _is_allowed_module_prefix(obj):
                        torch_object_ids[id(obj)] = f'{module.__name__}.{name}'
                        _find_torch_objects(obj)
                elif _is_allowed_module_prefix(obj):
                    if record:
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f'{module.__name__}.{name}'
                elif inspect.getmodule(obj) is None and (not is_safe_constant(obj)):
                    if record:
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f'{module.__name__}.{name}'
    _find_torch_objects(torch)
    _find_torch_objects(math)
    if config.trace_distributed:
        from torch.distributed import _functional_collectives_impl as fci
        for f in [fci._all_gather_into_tensor, fci._all_reduce, fci._reduce_scatter_tensor, fci._all_reduce_coalesced, fci._all_gather_into_tensor_coalesced, fci._reduce_scatter_tensor_coalesced]:
            torch_object_ids[id(f)] = repr(f)
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(method, (types.MethodDescriptorType, types.WrapperDescriptorType)):
            torch_object_ids[id(method)] = f'torch.Tensor.{name}'
    for idx in _disallowed_function_ids():
        if idx in torch_object_ids:
            del torch_object_ids[idx]
    for extra in (is_fx_tracing, is_compiling):
        torch_object_ids[id(extra)] = f'{extra.__module__}.{extra.__name__}'
    return AllowedObjects(torch_object_ids, ctx_mamager_classes, c_binding_in_graph_functions, non_c_binding_in_graph_functions, torch_name_rule_map)