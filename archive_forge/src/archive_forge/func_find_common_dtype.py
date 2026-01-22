import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def find_common_dtype(*xs):
    return _find_common_dtype(tuple(map(get_dtype_name, xs)), ())