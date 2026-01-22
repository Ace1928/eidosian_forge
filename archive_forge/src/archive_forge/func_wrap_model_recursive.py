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
def wrap_model_recursive(model: Model, wrapper: Callable[[Model], _ModelT]) -> _ModelT:
    """Recursively wrap a model and its submodules. The model is updated
    in-place."""
    for node in list(model.walk()):
        model.replace_node(node, wrapper(node))
    return wrapper(model)