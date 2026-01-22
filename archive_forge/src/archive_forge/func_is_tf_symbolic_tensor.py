import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def is_tf_symbolic_tensor(x):
    """
    Tests if `x` is a tensorflow symbolic tensor or not (ie. not eager). Safe to call even if tensorflow is not
    installed.
    """
    return False if not is_tf_available() else _is_tf_symbolic_tensor(x)