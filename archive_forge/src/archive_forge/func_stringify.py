import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def stringify(x, params):
    """Recursively stringify LazyArray instances in tuples, lists and dicts."""
    return _stringify_dispatch[x.__class__](x, params)