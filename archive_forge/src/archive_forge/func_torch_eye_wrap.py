import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_eye_wrap(fn):

    @functools.wraps(fn)
    def numpy_like(N, M=None, dtype=None, **kwargs):
        if dtype is not None:
            dtype = to_backend_dtype(dtype, like='torch')
        if M is not None:
            return fn(N, m=M, dtype=dtype, **kwargs)
        else:
            return fn(N, dtype=dtype, **kwargs)
    return numpy_like