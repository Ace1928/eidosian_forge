import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def infer_backend_multi(*arrays):
    """Infer which backend should be used for a function that takes multiple
    arguments. This assigns a priority to each backend, and returns the backend
    with the highest priority. By default, the priority is:

    - ``builtins``: -2
    - ``numpy``: -1
    - other backends: 0
    - ``autoray.lazy``: 1

    I.e. when mixing with ``numpy``, other array libraries are preferred, when
    mixing with ``autoray.lazy``, ``autoray.lazy`` is preferred. This has quite
    low overhead due to caching.
    """
    return _infer_class_backend_multi_cached(tuple((array.__class__ for array in arrays)))