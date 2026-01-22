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