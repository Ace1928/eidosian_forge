from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
def maybe_evaluate(value):
    """Evaluate value only if value is a :class:`lazy` instance."""
    if isinstance(value, lazy):
        return value.evaluate()
    return value