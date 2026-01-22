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
def create_script_class(obj):
    """
    Create and return a RecursiveScriptClass instance from a Python object.

    Arguments:
        obj: A Python object.
    """
    qualified_class_name = _jit_internal._qualified_name(type(obj))
    rcb = _jit_internal.createResolutionCallbackForClassMethods(type(obj))
    _compile_and_register_class(type(obj), rcb, qualified_class_name)
    class_ty = _python_cu.get_class(qualified_class_name)
    cpp_object = torch._C._create_object_with_type(class_ty)
    for name, value in obj.__dict__.items():
        cpp_object.setattr(name, value)
    return wrap_cpp_class(cpp_object)