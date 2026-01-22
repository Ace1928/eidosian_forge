import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def stringify_identity(x, params):
    if isinstance(x, (int, float, complex, bool, slice, range)):
        return f'{x}'
    if isinstance(x, str):
        return f"'{x}'"
    name = f'c{id(x)}'
    params.setdefault(name, x)
    return name