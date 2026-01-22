import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def choose_backend(fn, *args, like=None, **kwargs):
    """Choose a backend based on function name, arguments, and the ``like``
    keyword argument. The default, if ``like`` is not specified, is to infer
    the backend from the function call, the default of which is simply to use
    the first argument, if no custom dispatcher is found. Otherwise the
    backend is chosen based on the ``like`` argument - which can be an explicit
    backend name or an arbitrary object.
    """
    if like is None:
        return _infer_auto(fn, *args, **kwargs)
    elif isinstance(like, str):
        return like
    else:
        return infer_backend(like)