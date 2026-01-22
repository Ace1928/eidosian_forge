from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def load_store_chunk(x: Any, out: Any, index: slice, lock: Any, return_stored: bool, load_stored: bool):
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    x: array-like
        An array (potentially a NumPy one)
    out: array-like
        Where to store results.
    index: slice-like
        Where to store result from ``x`` in ``out``.
    lock: Lock-like or False
        Lock to use before writing to ``out``.
    return_stored: bool
        Whether to return ``out``.
    load_stored: bool
        Whether to return the array stored in ``out``.
        Ignored if ``return_stored`` is not ``True``.

    Returns
    -------

    If return_stored=True and load_stored=False
        out
    If return_stored=True and load_stored=True
        out[index]
    If return_stored=False and compute=False
        None

    Examples
    --------

    >>> a = np.ones((5, 6))
    >>> b = np.empty(a.shape)
    >>> load_store_chunk(a, b, (slice(None), slice(None)), False, False, False)
    """
    if lock:
        lock.acquire()
    try:
        if x is not None and x.size != 0:
            if is_arraylike(x):
                out[index] = x
            else:
                out[index] = np.asanyarray(x)
        if return_stored and load_stored:
            return out[index]
        elif return_stored and (not load_stored):
            return out
        else:
            return None
    finally:
        if lock:
            lock.release()