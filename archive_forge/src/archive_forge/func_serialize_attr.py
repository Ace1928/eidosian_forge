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
@functools.singledispatch
def serialize_attr(_: Any, value: Any, name: str, model: Model) -> bytes:
    """Serialize an attribute value (defaults to msgpack). You can register
    custom serializers using the @serialize_attr.register decorator with the
    type to serialize, e.g.: @serialize_attr.register(MyCustomObject).
    """
    return srsly.msgpack_dumps(value)