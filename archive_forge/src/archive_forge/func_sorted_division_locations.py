from __future__ import annotations
from collections.abc import Iterable
from functools import partial
from math import ceil
from operator import getitem
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
import pandas as pd
import dask.array as da
from dask.base import is_dask_collection, tokenize
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.dataframe._compat import is_any_real_numeric_dtype
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import (
from dask.dataframe.dispatch import meta_lib_from_array
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import M, funcname, is_arraylike
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    """Find division locations and values in sorted list

    Examples
    --------

    >>> L = ['A', 'B', 'C', 'D', 'E', 'F']
    >>> sorted_division_locations(L, chunksize=2)
    (['A', 'C', 'E', 'F'], [0, 2, 4, 6])

    >>> sorted_division_locations(L, chunksize=3)
    (['A', 'D', 'F'], [0, 3, 6])

    >>> L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
    >>> sorted_division_locations(L, chunksize=3)
    (['A', 'B', 'C', 'C'], [0, 4, 7, 8])

    >>> sorted_division_locations(L, chunksize=2)
    (['A', 'B', 'C', 'C'], [0, 4, 7, 8])

    >>> sorted_division_locations(['A'], chunksize=2)
    (['A', 'A'], [0, 1])
    """
    if (npartitions is None) == (chunksize is None):
        raise ValueError('Exactly one of npartitions and chunksize must be specified.')
    seq_unique = seq.unique() if hasattr(seq, 'unique') else np.unique(seq)
    duplicates = len(seq_unique) < len(seq)
    enforce_exact = False
    if duplicates:
        offsets = seq.searchsorted(seq_unique, side='left') if hasattr(seq, 'searchsorted') else np.array(seq).searchsorted(seq_unique, side='left')
        enforce_exact = npartitions and len(offsets) >= npartitions
    else:
        offsets = seq_unique = None
    residual = 0
    subtract_drift = False
    if npartitions:
        chunksize = len(seq) // npartitions
        residual = len(seq) % npartitions
        subtract_drift = True

    def chunksizes(ind):
        return chunksize + int(ind < residual)
    divisions = [seq[0]]
    locations = [0]
    i = chunksizes(0)
    ind = None
    drift = 0
    divs_remain = npartitions - len(divisions) if enforce_exact else None
    while i < len(seq):
        div = seq[i]
        if duplicates:
            if ind is None:
                ind = int((seq_unique == seq[i]).nonzero()[0][0])
            if enforce_exact:
                offs_remain = len(offsets) - ind
                if divs_remain > offs_remain:
                    ind -= divs_remain - offs_remain
                    i = offsets[ind]
                    div = seq[i]
            pos = int(offsets[ind])
        else:
            pos = i
        if div <= divisions[-1]:
            if duplicates:
                ind += 1
                i = int(offsets[ind]) if ind < len(offsets) else len(seq)
            else:
                i += 1
        else:
            if subtract_drift:
                drift = drift + (pos - locations[-1] - chunksizes(len(divisions) - 1))
            if enforce_exact:
                divs_remain -= 1
            i = pos + max(1, chunksizes(len(divisions)) - drift)
            divisions.append(div)
            locations.append(pos)
            ind = None
    divisions.append(seq[-1])
    locations.append(len(seq))
    return (divisions, locations)