from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def safe_sqrt(a):
    """A version of sqrt that properly handles scalar masked arrays.

    To mimic ``np.ma`` reductions, we need to convert scalar masked arrays that
    have an active mask to the ``np.ma.masked`` singleton. This is properly
    handled automatically for reduction code, but not for ufuncs. We implement
    a simple version here, since calling `np.ma.sqrt` everywhere is
    significantly more expensive.
    """
    if hasattr(a, '_elemwise'):
        return a._elemwise(_sqrt, a)
    return _sqrt(a)