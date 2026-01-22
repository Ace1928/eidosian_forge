from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from
Map a function over blocks of arrays with some overlap

    We share neighboring zones between blocks of the array, map a
    function, and then trim away the neighboring strips. If depth is
    larger than any chunk along a particular axis, then the array is
    rechunked.

    Note that this function will attempt to automatically determine the output
    array type before computing it, please refer to the ``meta`` keyword argument
    in ``map_blocks`` if you expect that the function will not succeed when
    operating on 0-d arrays.

    Parameters
    ----------
    func: function
        The function to apply to each extended block.
        If multiple arrays are provided, then the function should expect to
        receive chunks of each array in the same order.
    args : dask arrays
    depth: int, tuple, dict or list, keyword only
        The number of elements that each block should share with its neighbors
        If a tuple or dict then this can be different per axis.
        If a list then each element of that list must be an int, tuple or dict
        defining depth for the corresponding array in `args`.
        Asymmetric depths may be specified using a dict value of (-/+) tuples.
        Note that asymmetric depths are currently only supported when
        ``boundary`` is 'none'.
        The default value is 0.
    boundary: str, tuple, dict or list, keyword only
        How to handle the boundaries.
        Values include 'reflect', 'periodic', 'nearest', 'none',
        or any constant value like 0 or np.nan.
        If a list then each element must be a str, tuple or dict defining the
        boundary for the corresponding array in `args`.
        The default value is 'reflect'.
    trim: bool, keyword only
        Whether or not to trim ``depth`` elements from each block after
        calling the map function.
        Set this to False if your mapping function already does this for you
    align_arrays: bool, keyword only
        Whether or not to align chunks along equally sized dimensions when
        multiple arrays are provided.  This allows for larger chunks in some
        arrays to be broken into smaller ones that match chunk sizes in other
        arrays such that they are compatible for block function mapping. If
        this is false, then an error will be thrown if arrays do not already
        have the same number of blocks in each dimension.
    allow_rechunk: bool, keyword only
        Allows rechunking, otherwise chunk sizes need to match and core
        dimensions are to consist only of one chunk.
    **kwargs:
        Other keyword arguments valid in ``map_blocks``

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da

    >>> x = np.array([1, 1, 2, 3, 3, 3, 2, 1, 1])
    >>> x = da.from_array(x, chunks=5)
    >>> def derivative(x):
    ...     return x - np.roll(x, 1)

    >>> y = x.map_overlap(derivative, depth=1, boundary=0)
    >>> y.compute()
    array([ 1,  0,  1,  1,  0,  0, -1, -1,  0])

    >>> x = np.arange(16).reshape((4, 4))
    >>> d = da.from_array(x, chunks=(2, 2))
    >>> d.map_overlap(lambda x: x + x.size, depth=1, boundary='reflect').compute()
    array([[16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])

    >>> func = lambda x: x + x.size
    >>> depth = {0: 1, 1: 1}
    >>> boundary = {0: 'reflect', 1: 'none'}
    >>> d.map_overlap(func, depth, boundary).compute()  # doctest: +NORMALIZE_WHITESPACE
    array([[12,  13,  14,  15],
           [16,  17,  18,  19],
           [20,  21,  22,  23],
           [24,  25,  26,  27]])

    The ``da.map_overlap`` function can also accept multiple arrays.

    >>> func = lambda x, y: x + y
    >>> x = da.arange(8).reshape(2, 4).rechunk((1, 2))
    >>> y = da.arange(4).rechunk(2)
    >>> da.map_overlap(func, x, y, depth=1, boundary='reflect').compute() # doctest: +NORMALIZE_WHITESPACE
    array([[ 0,  2,  4,  6],
           [ 4,  6,  8,  10]])

    When multiple arrays are given, they do not need to have the
    same number of dimensions but they must broadcast together.
    Arrays are aligned block by block (just as in ``da.map_blocks``)
    so the blocks must have a common chunk size.  This common chunking
    is determined automatically as long as ``align_arrays`` is True.

    >>> x = da.arange(8, chunks=4)
    >>> y = da.arange(8, chunks=2)
    >>> r = da.map_overlap(func, x, y, depth=1, boundary='reflect', align_arrays=True)
    >>> len(r.to_delayed())
    4

    >>> da.map_overlap(func, x, y, depth=1, boundary='reflect', align_arrays=False).compute()
    Traceback (most recent call last):
        ...
    ValueError: Shapes do not align {'.0': {2, 4}}

    Note also that this function is equivalent to ``map_blocks``
    by default.  A non-zero ``depth`` must be defined for any
    overlap to appear in the arrays provided to ``func``.

    >>> func = lambda x: x.sum()
    >>> x = da.ones(10, dtype='int')
    >>> block_args = dict(chunks=(), drop_axis=0)
    >>> da.map_blocks(func, x, **block_args).compute()
    10
    >>> da.map_overlap(func, x, **block_args, boundary='reflect').compute()
    10
    >>> da.map_overlap(func, x, **block_args, depth=1, boundary='reflect').compute()
    12

    For functions that may not handle 0-d arrays, it's also possible to specify
    ``meta`` with an empty array matching the type of the expected result. In
    the example below, ``func`` will result in an ``IndexError`` when computing
    ``meta``:

    >>> x = np.arange(16).reshape((4, 4))
    >>> d = da.from_array(x, chunks=(2, 2))
    >>> y = d.map_overlap(lambda x: x + x[2], depth=1, boundary='reflect', meta=np.array(()))
    >>> y
    dask.array<_trim, shape=(4, 4), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>
    >>> y.compute()
    array([[ 4,  6,  8, 10],
           [ 8, 10, 12, 14],
           [20, 22, 24, 26],
           [24, 26, 28, 30]])

    Similarly, it's possible to specify a non-NumPy array to ``meta``:

    >>> import cupy  # doctest: +SKIP
    >>> x = cupy.arange(16).reshape((4, 4))  # doctest: +SKIP
    >>> d = da.from_array(x, chunks=(2, 2))  # doctest: +SKIP
    >>> y = d.map_overlap(lambda x: x + x[2], depth=1, boundary='reflect', meta=cupy.array(()))  # doctest: +SKIP
    >>> y  # doctest: +SKIP
    dask.array<_trim, shape=(4, 4), dtype=float64, chunksize=(2, 2), chunktype=cupy.ndarray>
    >>> y.compute()  # doctest: +SKIP
    array([[ 4,  6,  8, 10],
           [ 8, 10, 12, 14],
           [20, 22, 24, 26],
           [24, 26, 28, 30]])
    