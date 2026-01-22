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
def prefixscan_blelloch(func, preop, binop, x, axis=None, dtype=None, out=None):
    """Generic function to perform parallel cumulative scan (a.k.a prefix scan)

    The Blelloch prefix scan is work-efficient and exposes parallelism.
    A parallel cumsum works by first taking the sum of each block, then do a binary tree
    merge followed by a fan-out (i.e., the Brent-Kung pattern).  We then take the cumsum
    of each block and add the sum of the previous blocks.

    When performing a cumsum across N chunks, this method has 2 * lg(N) levels of dependencies.
    In contrast, the sequential method has N levels of dependencies.

    Floating point operations should be more accurate with this method compared to sequential.

    Parameters
    ----------
    func : callable
        Cumulative function (e.g. ``np.cumsum``)
    preop : callable
        Function to get the final value of a cumulative function (e.g., ``np.sum``)
    binop : callable
        Associative function (e.g. ``add``)
    x : dask array
    axis : int
    dtype : dtype

    Returns
    -------
    dask array
    """
    if axis is None:
        x = x.flatten().rechunk(chunks=x.npartitions)
        axis = 0
    if dtype is None:
        dtype = getattr(func(np.ones((0,), dtype=x.dtype)), 'dtype', object)
    assert isinstance(axis, Integral)
    axis = validate_axis(axis, x.ndim)
    name = f'{func.__name__}-{tokenize(func, axis, preop, binop, x, dtype)}'
    base_key = (name,)
    batches = x.map_blocks(preop, axis=axis, keepdims=True, dtype=dtype)
    *indices, last_index = full_indices = [list(product(*[range(nb) if j != axis else [i] for j, nb in enumerate(x.numblocks)])) for i in range(x.numblocks[axis])]
    prefix_vals = [[(batches.name,) + index for index in vals] for vals in indices]
    dsk = {}
    n_vals = len(prefix_vals)
    level = 0
    if n_vals >= 2:
        stride = 1
        stride2 = 2
        while stride2 <= n_vals:
            for i in range(stride2 - 1, n_vals, stride2):
                new_vals = []
                for index, left_val, right_val in zip(indices[i], prefix_vals[i - stride], prefix_vals[i]):
                    key = base_key + index + (level, i)
                    dsk[key] = (binop, left_val, right_val)
                    new_vals.append(key)
                prefix_vals[i] = new_vals
            stride = stride2
            stride2 *= 2
            level += 1
        stride2 = builtins.max(2, 2 ** math.ceil(math.log2(n_vals // 2)))
        stride = stride2 // 2
        while stride > 0:
            for i in range(stride2 + stride - 1, n_vals, stride2):
                new_vals = []
                for index, left_val, right_val in zip(indices[i], prefix_vals[i - stride], prefix_vals[i]):
                    key = base_key + index + (level, i)
                    dsk[key] = (binop, left_val, right_val)
                    new_vals.append(key)
                prefix_vals[i] = new_vals
            stride2 = stride
            stride //= 2
            level += 1
    if full_indices:
        for index in full_indices[0]:
            dsk[base_key + index] = (_prefixscan_first, func, (x.name,) + index, axis, dtype)
        for indexes, vals in zip(drop(1, full_indices), prefix_vals):
            for index, val in zip(indexes, vals):
                dsk[base_key + index] = (_prefixscan_combine, func, binop, val, (x.name,) + index, axis, dtype)
    if len(full_indices) < 2:
        deps = [x]
    else:
        deps = [x, batches]
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    result = Array(graph, name, x.chunks, batches.dtype)
    return handle_out(out, result)