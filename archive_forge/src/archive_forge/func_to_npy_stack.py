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
def to_npy_stack(dirname, x, axis=0):
    """Write dask array to a stack of .npy files

    This partitions the dask.array along one axis and stores each block along
    that axis as a single .npy file in the specified directory

    Examples
    --------
    >>> x = da.ones((5, 10, 10), chunks=(2, 4, 4))  # doctest: +SKIP
    >>> da.to_npy_stack('data/', x, axis=0)  # doctest: +SKIP

    The ``.npy`` files store numpy arrays for ``x[0:2], x[2:4], and x[4:5]``
    respectively, as is specified by the chunk size along the zeroth axis::

        $ tree data/
        data/
        |-- 0.npy
        |-- 1.npy
        |-- 2.npy
        |-- info

    The ``info`` file stores the dtype, chunks, and axis information of the array.
    You can load these stacks with the :func:`dask.array.from_npy_stack` function.

    >>> y = da.from_npy_stack('data/')  # doctest: +SKIP

    See Also
    --------
    from_npy_stack
    """
    chunks = tuple((c if i == axis else (sum(c),) for i, c in enumerate(x.chunks)))
    xx = x.rechunk(chunks)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    meta = {'chunks': chunks, 'dtype': x.dtype, 'axis': axis}
    with open(os.path.join(dirname, 'info'), 'wb') as f:
        pickle.dump(meta, f)
    name = 'to-npy-stack-' + str(uuid.uuid1())
    dsk = {(name, i): (np.save, os.path.join(dirname, '%d.npy' % i), key) for i, key in enumerate(core.flatten(xx.__dask_keys__()))}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[xx])
    compute_as_if_collection(Array, graph, list(dsk))