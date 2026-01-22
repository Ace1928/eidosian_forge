import contextlib
import copy
import functools
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import (
import srsly
from .backends import CupyOps, NumpyOps, Ops, ParamServer, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .types import FloatsXd
from .util import (
@property
def grad_names(self) -> Tuple[str, ...]:
    """Get the names of parameters with registered gradients (including unset)."""
    return tuple([name for name in self.param_names if self.has_grad(name)])