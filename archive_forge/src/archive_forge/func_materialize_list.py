import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def materialize_list(x):
    return list(map(maybe_materialize, x))