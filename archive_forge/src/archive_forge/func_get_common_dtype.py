import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def get_common_dtype(*arrays):
    """Compute the minimal dtype sufficient for ``arrays``."""
    dtypes = set(map(get_dtype_name, arrays))
    has_complex = not _COMPLEX_DTYPES.isdisjoint(dtypes)
    has_double = not _DOUBLE_DTYPES.isdisjoint(dtypes)
    return _DTYPE_MAP[has_complex, has_double]