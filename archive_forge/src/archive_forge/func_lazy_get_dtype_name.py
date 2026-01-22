import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def lazy_get_dtype_name(x):
    return x.dtype