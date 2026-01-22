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
def insert_to_ooc(keys: list, chunks: tuple[tuple[int, ...], ...], out: ArrayLike, name: str, *, lock: Lock | bool=True, region: tuple[slice, ...] | slice | None=None, return_stored: bool=False, load_stored: bool=False) -> dict:
    """
    Creates a Dask graph for storing chunks from ``arr`` in ``out``.

    Parameters
    ----------
    keys: list
        Dask keys of the input array
    chunks: tuple
        Dask chunks of the input array
    out: array-like
        Where to store results to
    name: str
        First element of dask keys
    lock: Lock-like or bool, optional
        Whether to lock or with what (default is ``True``,
        which means a :class:`threading.Lock` instance).
    region: slice-like, optional
        Where in ``out`` to store ``arr``'s results
        (default is ``None``, meaning all of ``out``).
    return_stored: bool, optional
        Whether to return ``out``
        (default is ``False``, meaning ``None`` is returned).
    load_stored: bool, optional
        Whether to handling loading from ``out`` at the same time.
        Ignored if ``return_stored`` is not ``True``.
        (default is ``False``, meaning defer to ``return_stored``).

    Returns
    -------
    dask graph of store operation

    Examples
    --------
    >>> import dask.array as da
    >>> d = da.ones((5, 6), chunks=(2, 3))
    >>> a = np.empty(d.shape)
    >>> insert_to_ooc(d.__dask_keys__(), d.chunks, a, "store-123")  # doctest: +SKIP
    """
    if lock is True:
        lock = Lock()
    slices = slices_from_chunks(chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]
    if return_stored and load_stored:
        func = load_store_chunk
        args = (load_stored,)
    else:
        func = store_chunk
        args = ()
    dsk = {(name,) + t[1:]: (func, t, out, slc, lock, return_stored) + args for t, slc in zip(core.flatten(keys), slices)}
    return dsk