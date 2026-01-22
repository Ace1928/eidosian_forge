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
def trim_overlap(x, depth, boundary=None):
    """Trim sides from each block.

    This couples well with the ``map_overlap`` operation which may leave
    excess data on each block.

    See also
    --------
    dask.array.overlap.map_overlap

    """
    axes = coerce_depth(x.ndim, depth)
    return trim_internal(x, axes=axes, boundary=boundary)