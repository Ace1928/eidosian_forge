import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def getitem_hasher(_, a, key):
    if not isinstance(key, tuple):
        key = (key,)
    hkey = tuple((str(k) if isinstance(k, slice) else id(k) if hasattr(k, 'shape') else k for k in key))
    return f'getitem-{hash((id(a), hkey))}'