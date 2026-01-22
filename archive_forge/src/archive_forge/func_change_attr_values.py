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
def change_attr_values(model: _ModelT, mapping: Dict[str, Dict[str, Any]]) -> _ModelT:
    """Walk over the model's nodes, changing the value of attributes using the
    provided mapping, which maps node names to attr names to attr values.
    """
    for node in model.walk():
        if node.name in mapping:
            attrs = mapping[node.name]
            for attr, value in attrs.items():
                if attr in node.attrs:
                    node.attrs[attr] = value
    return model