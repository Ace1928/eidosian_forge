import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def register_dispatch(fun, dispatcher):
    """Register a new dispatcher.

    This is useful in case the backend to be used by a function cannot be
    inferred from the first argument.
    """
    _DISPATCHERS[fun] = dispatcher