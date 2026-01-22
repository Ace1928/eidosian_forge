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
def maybe_get_ref(self, name: str) -> Optional['Model']:
    """Retrieve the value of a reference if it exists, or None."""
    return self.get_ref(name) if self.has_ref(name) else None