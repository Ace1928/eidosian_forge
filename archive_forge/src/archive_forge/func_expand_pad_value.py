from __future__ import annotations
import itertools
from collections.abc import Sequence
from functools import partial
from itertools import product
from numbers import Integral, Number
import numpy as np
from tlz import sliding_window
from dask.array import chunk
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.numpy_compat import AxisError
from dask.array.ufunc import greater_equal, rint
from dask.array.utils import meta_from_array
from dask.array.wrap import empty, full, ones, zeros
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, derived_from, is_cupy_type
def expand_pad_value(array, pad_value):
    if isinstance(pad_value, Number) or getattr(pad_value, 'ndim', None) == 0:
        pad_value = array.ndim * ((pad_value, pad_value),)
    elif isinstance(pad_value, Sequence) and all((isinstance(pw, Number) for pw in pad_value)) and (len(pad_value) == 1):
        pad_value = array.ndim * ((pad_value[0], pad_value[0]),)
    elif isinstance(pad_value, Sequence) and len(pad_value) == 2 and all((isinstance(pw, Number) for pw in pad_value)):
        pad_value = array.ndim * (tuple(pad_value),)
    elif isinstance(pad_value, Sequence) and len(pad_value) == array.ndim and all((isinstance(pw, Sequence) for pw in pad_value)) and all((len(pw) == 2 for pw in pad_value)) and all((all((isinstance(w, Number) for w in pw)) for pw in pad_value)):
        pad_value = tuple((tuple(pw) for pw in pad_value))
    elif isinstance(pad_value, Sequence) and len(pad_value) == 1 and isinstance(pad_value[0], Sequence) and (len(pad_value[0]) == 2) and all((isinstance(pw, Number) for pw in pad_value[0])):
        pad_value = array.ndim * (tuple(pad_value[0]),)
    else:
        raise TypeError('`pad_value` must be composed of integral typed values.')
    return pad_value