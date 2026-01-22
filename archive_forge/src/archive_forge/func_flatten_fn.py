import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
def flatten_fn(*args):
    tree_args = pytree.tree_unflatten(list(args), in_spec)
    tree_out = root_fn(*tree_args)
    out_args, out_spec = pytree.tree_flatten(tree_out)
    assert isinstance(self.graph._codegen, _PyTreeCodeGen)
    self.graph._codegen.pytree_info = self.graph._codegen.pytree_info._replace(out_spec=out_spec)
    return out_args