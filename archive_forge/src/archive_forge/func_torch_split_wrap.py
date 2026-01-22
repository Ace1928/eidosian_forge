import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_split_wrap(fn):

    @functools.wraps(fn)
    def numpy_like(ary, indices_or_sections, axis=0, **kwargs):
        if isinstance(indices_or_sections, int):
            split_size = shape(ary)[axis] // indices_or_sections
            return fn(ary, split_size, dim=axis, **kwargs)
        else:
            if len(indices_or_sections) == 0:
                return (ary,)
            diff = do('diff', indices_or_sections, prepend=0, append=shape(ary)[axis], like='numpy')
            diff = list(diff)
            return fn(ary, diff, dim=axis)
    return numpy_like