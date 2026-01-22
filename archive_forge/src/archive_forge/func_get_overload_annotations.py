import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def get_overload_annotations(mod, jit_ignored_properties):
    overloads = {}
    for name in dir(type(mod)):
        if name in jit_ignored_properties:
            continue
        item = getattr(mod, name, None)
        if not callable(item):
            continue
        if hasattr(item, '__module__') and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(item, mod.__class__)
            if method_overloads is None:
                continue
            if item.__func__ in method_overloads:
                raise RuntimeError(_jit_internal.get_overload_no_implementation_error_message('method', item.__func__))
            names = [name + '__' + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))
    return overloads