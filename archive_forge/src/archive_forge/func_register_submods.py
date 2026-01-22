import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def register_submods(mod, prefix):
    for name, child in mod.named_children():
        submod_qualname = prefix + '.' + name
        trace_module_map[child] = submod_qualname
        register_submods(child, submod_qualname)