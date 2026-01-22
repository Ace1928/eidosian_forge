from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
def slicing_plan(chunks, index):
    """Construct a plan to slice chunks with the given index

    Parameters
    ----------
    chunks : Tuple[int]
        One dimensions worth of chunking information
    index : np.ndarray[int]
        The index passed to slice on that dimension

    Returns
    -------
    out : List[Tuple[int, np.ndarray]]
        A list of chunk/sub-index pairs corresponding to each output chunk
    """
    from dask.array.utils import asarray_safe
    if not is_arraylike(index):
        index = np.asanyarray(index)
    cum_chunks_tup = cached_cumsum(chunks)
    cum_chunks = asarray_safe(cum_chunks_tup, like=index)
    if cum_chunks.dtype.kind != 'f':
        cum_chunks = cum_chunks.astype(index.dtype)
    chunk_locations = np.searchsorted(cum_chunks, index, side='right')
    chunk_locations = chunk_locations.tolist()
    where = np.where(np.diff(chunk_locations))[0] + 1
    extra = asarray_safe([0], like=where)
    c_loc = asarray_safe([len(chunk_locations)], like=where)
    where = np.concatenate([extra, where, c_loc])
    out = []
    for i in range(len(where) - 1):
        sub_index = index[where[i]:where[i + 1]]
        chunk = chunk_locations[where[i]]
        if chunk > 0:
            sub_index = sub_index - cum_chunks[chunk - 1]
        out.append((chunk, sub_index))
    return out