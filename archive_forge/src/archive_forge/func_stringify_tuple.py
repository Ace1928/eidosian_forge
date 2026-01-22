import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def stringify_tuple(x, params):
    if not x:
        return '()'
    return f'({', '.join((stringify(xi, params) for xi in x))},)'