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
def compute_chunk_sizes(self):
    """
        Compute the chunk sizes for a Dask array. This is especially useful
        when the chunk sizes are unknown (e.g., when indexing one Dask array
        with another).

        Notes
        -----
        This function modifies the Dask array in-place.

        Examples
        --------
        >>> import dask.array as da
        >>> import numpy as np
        >>> x = da.from_array([-2, -1, 0, 1, 2], chunks=2)
        >>> x.chunks
        ((2, 2, 1),)
        >>> y = x[x <= 0]
        >>> y.chunks
        ((nan, nan, nan),)
        >>> y.compute_chunk_sizes()  # in-place computation
        dask.array<getitem, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
        >>> y.chunks
        ((2, 1, 0),)

        """
    x = self
    chunk_shapes = x.map_blocks(_get_chunk_shape, dtype=int, chunks=tuple((len(c) * (1,) for c in x.chunks)) + ((x.ndim,),), new_axis=x.ndim)
    c = []
    for i in range(x.ndim):
        s = x.ndim * [0] + [i]
        s[i] = slice(None)
        s = tuple(s)
        c.append(tuple(chunk_shapes[s]))
    x._chunks = tuple((tuple((int(chunk) for chunk in chunks)) for chunks in compute(tuple(c))[0]))
    return x