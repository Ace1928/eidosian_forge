import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tensorflow_split_wrap(fn):

    @functools.wraps(fn)
    def numpy_like(ary, indices_or_sections, axis=0, **kwargs):
        if isinstance(indices_or_sections, int):
            return fn(ary, indices_or_sections, axis=axis, **kwargs)
        else:
            diff = do('diff', indices_or_sections, prepend=0, append=shape(ary)[axis], like='numpy')
            diff = list(diff)
            return fn(ary, diff, axis=axis)
    return numpy_like