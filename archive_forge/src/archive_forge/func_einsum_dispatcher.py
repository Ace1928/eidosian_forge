import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def einsum_dispatcher(*args, **_):
    """Dispatcher for handling einsum.

    einsum can be called with a str equation as the first argument, or with
    'interleaved' inputs. This dispatcher handles both cases and also takes
    into account all arrays.
    """
    return infer_backend_multi(*args)