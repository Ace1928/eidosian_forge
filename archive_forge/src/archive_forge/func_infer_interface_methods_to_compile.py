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
def infer_interface_methods_to_compile(nn_module):
    """Rule to infer the methods from the interface type.

        It is used to know which methods need to act as starting points for compilation.
        """
    stubs = []
    for method in mod_interface.getMethodNames():
        stubs.append(make_stub_from_method(nn_module, method))
    return stubs