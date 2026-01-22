import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
@functools.wraps(original_init)
def init_then_script(self, *args, **kwargs):
    num_methods = len(cls._methods)
    original_init(self, *args, **kwargs)
    added_methods_in_init = len(cls._methods) > num_methods
    if type(self) == cls:

        def make_stubs(module):
            cls = type(module)
            if hasattr(cls, '_methods'):
                return [v for k, v in sorted(cls._methods.items())]
            else:
                return infer_methods_to_compile(module)
        self.__dict__['_actual_script_module'] = torch.jit._recursive.create_script_module(self, make_stubs, share_types=not added_methods_in_init)
        concrete_type = self._actual_script_module._concrete_type
        for name in concrete_type.get_attributes():
            delattr(self, name)
        for name, _ in concrete_type.get_modules():
            delattr(self, name)
        for name in ('_parameters', '_buffers', '_modules'):
            delattr(self, name)