import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def stringify_dict(x, params):
    entries = (f'{k}: {stringify(v, params)}' for k, v in x.items())
    return f'{{{', '.join(entries)}}}'