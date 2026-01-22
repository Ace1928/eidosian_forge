import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def is_lazy_array(x):
    """Check if ``x`` is a lazy array."""
    return isinstance(x, LazyArray)