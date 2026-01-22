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
def maybe_warn_nondeterministic():
    if has_warned[0]:
        return
    has_warned[0] = True
    nondeterm_ops = [op for op in traced_func.graph.nodes() if op.isNondeterministic()]
    if len(nondeterm_ops) > 0:
        nondeterministic_ops_warning = 'Trace had nondeterministic nodes. '
        nondeterministic_ops_warning += 'Did you forget call .eval() on your model? Nodes:\n'
        nondeterministic_ops_warning += '\n'.join([indent(str(op)) for op in nondeterm_ops][:20])
        nondeterministic_ops_warning += '\nThis may cause errors in trace checking. To disable trace checking, pass check_trace=False to torch.jit.trace()'
        warnings.warn(nondeterministic_ops_warning, category=TracerWarning, stacklevel=5)