import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@lazy_cache(name)
def binary_func(x1, x2):
    x1shape = getattr(x1, 'shape', ())
    x2shape = getattr(x2, 'shape', ())
    newshape = find_broadcast_shape(x1shape, x2shape)
    return LazyArray(backend=find_common_backend(x1, x2), fn=fn, args=(x1, x2), kwargs=None, shape=newshape, deps=tuple((x for x in (x1, x2) if isinstance(x, LazyArray))))