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
@insert_meta_param_description
def from_delayed(dfs: Delayed | distributed.Future | Iterable[Delayed | distributed.Future], meta=None, divisions: tuple | Literal['sorted'] | None=None, prefix: str='from-delayed', verify_meta: bool=True) -> DataFrame | Series:
    """Create Dask DataFrame from many Dask Delayed objects

    Parameters
    ----------
    dfs :
        A ``dask.delayed.Delayed``, a ``distributed.Future``, or an iterable of either
        of these objects, e.g. returned by ``client.submit``. These comprise the
        individual partitions of the resulting dataframe.
        If a single object is provided (not an iterable), then the resulting dataframe
        will have only one partition.
    $META
    divisions :
        Partition boundaries along the index.
        For tuple, see https://docs.dask.org/en/latest/dataframe-design.html#partitions
        For string 'sorted' will compute the delayed values to find index
        values.  Assumes that the indexes are mutually sorted.
        If None, then won't use index information
    prefix :
        Prefix to prepend to the keys.
    verify_meta :
        If True check that the partitions have consistent metadata, defaults to True.
    """
    from dask.delayed import Delayed
    if isinstance(dfs, Delayed) or hasattr(dfs, 'key'):
        dfs = [dfs]
    dfs = [delayed(df) if not isinstance(df, Delayed) and hasattr(df, 'key') else df for df in dfs]
    for item in dfs:
        if not isinstance(item, Delayed):
            raise TypeError('Expected Delayed object, got %s' % type(item).__name__)
    if meta is None:
        meta = delayed(make_meta)(dfs[0]).compute()
    else:
        meta = make_meta(meta)
    if not dfs:
        dfs = [delayed(make_meta)(meta)]
    if divisions is None or divisions == 'sorted':
        divs: list | tuple = [None] * (len(dfs) + 1)
    else:
        divs = list(divisions)
        if len(divs) != len(dfs) + 1:
            raise ValueError('divisions should be a tuple of len(dfs) + 1')
    name = prefix + '-' + tokenize(*dfs)
    layer = DataFrameIOLayer(name=name, columns=None, inputs=BlockwiseDepDict({(i,): inp.key for i, inp in enumerate(dfs)}, produces_keys=True), io_func=partial(check_meta, meta=meta, funcname='from_delayed') if verify_meta else lambda x: x)
    df = new_dd_object(HighLevelGraph.from_collections(name, layer, dfs), name, meta, divs)
    if divisions == 'sorted':
        from dask.dataframe.shuffle import compute_and_set_divisions
        return compute_and_set_divisions(df)
    return df