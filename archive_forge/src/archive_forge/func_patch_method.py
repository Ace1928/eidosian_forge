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
def patch_method(self, cls: type, name: str, new_fn: Callable, deduplicate: bool=True):
    """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
    new_fn.__fx_already_patched = deduplicate
    orig_fn = getattr(cls, name)
    if getattr(orig_fn, '__fx_already_patched', False):
        return
    self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
    setattr(cls, name, new_fn)