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
def pad_stats(array, pad_width, mode, stat_length):
    """
    Helper function for padding boundaries with statistics from the array.

    In cases where the padding requires computations of statistics from part
    or all of the array, this function helps compute those statistics as
    requested and then adds those statistics onto the boundaries of the array.
    """
    if mode == 'median':
        raise NotImplementedError('`pad` does not support `mode` of `median`.')
    stat_length = expand_pad_value(array, stat_length)
    result = np.empty(array.ndim * (3,), dtype=object)
    for idx in np.ndindex(result.shape):
        axes = []
        select = []
        pad_shape = []
        pad_chunks = []
        for d, (i, s, c, w, l) in enumerate(zip(idx, array.shape, array.chunks, pad_width, stat_length)):
            if i < 1:
                axes.append(d)
                select.append(slice(None, l[0], None))
                pad_shape.append(w[0])
                pad_chunks.append(w[0])
            elif i > 1:
                axes.append(d)
                select.append(slice(s - l[1], None, None))
                pad_shape.append(w[1])
                pad_chunks.append(w[1])
            else:
                select.append(slice(None))
                pad_shape.append(s)
                pad_chunks.append(c)
        axes = tuple(axes)
        select = tuple(select)
        pad_shape = tuple(pad_shape)
        pad_chunks = tuple(pad_chunks)
        result_idx = array[select]
        if axes:
            if mode == 'maximum':
                result_idx = result_idx.max(axis=axes, keepdims=True)
            elif mode == 'mean':
                result_idx = result_idx.mean(axis=axes, keepdims=True)
            elif mode == 'minimum':
                result_idx = result_idx.min(axis=axes, keepdims=True)
            result_idx = broadcast_to(result_idx, pad_shape, chunks=pad_chunks)
            if mode == 'mean':
                if np.issubdtype(array.dtype, np.integer):
                    result_idx = rint(result_idx)
                result_idx = result_idx.astype(array.dtype)
        result[idx] = result_idx
    result = block(result.tolist())
    return result