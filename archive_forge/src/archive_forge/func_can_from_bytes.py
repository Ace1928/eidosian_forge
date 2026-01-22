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
def can_from_bytes(self, bytes_data: bytes, *, strict: bool=True) -> bool:
    """Check whether the bytes data is compatible with the model. If 'strict',
        the function returns False if the model has an attribute already loaded
        that would be changed.
        """
    try:
        msg = srsly.msgpack_loads(bytes_data)
    except ValueError:
        return False
    return self.can_from_dict(msg, strict=strict)