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
def interface_script(mod_interface, nn_module):
    """
    Make a ScriptModule from an nn.Module, using the interface methods rule for determining which methods to compile.

    Args:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module
    check_module_initialized(nn_module)

    def infer_interface_methods_to_compile(nn_module):
        """Rule to infer the methods from the interface type.

        It is used to know which methods need to act as starting points for compilation.
        """
        stubs = []
        for method in mod_interface.getMethodNames():
            stubs.append(make_stub_from_method(nn_module, method))
        return stubs
    return create_script_module(nn_module, infer_interface_methods_to_compile)