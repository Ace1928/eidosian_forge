from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def stack_partitions(dfs, divisions, join='outer', ignore_order=False, **kwargs):
    """Concatenate partitions on axis=0 by doing a simple stack"""
    kwargs.update({'ignore_order': ignore_order})
    meta = make_meta(methods.concat([df._meta_nonempty for df in dfs if not is_dataframe_like(df) or len(df._meta_nonempty.columns) > 0], join=join, filter_warning=False, **kwargs))
    empty = strip_unknown_categories(meta)
    name = f'concat-{tokenize(*dfs)}'
    dsk = {}
    i = 0
    astyped_dfs = []
    for df in dfs:
        if is_dataframe_like(df):
            shared_columns = df.columns.intersection(meta.columns)
            needs_astype = [col for col in shared_columns if df[col].dtype != meta[col].dtype and (not isinstance(df[col].dtype, pd.CategoricalDtype))]
            if needs_astype:
                df = df.copy()
                df[needs_astype] = df[needs_astype].astype(meta[needs_astype].dtypes)
        if is_series_like(df) and is_series_like(meta):
            if not df.dtype == meta.dtype and (not isinstance(df.dtype, pd.CategoricalDtype)):
                df = df.astype(meta.dtype)
        else:
            pass
        astyped_dfs.append(df)
        try:
            check_meta(df._meta, meta)
            match = True
        except (ValueError, TypeError):
            match = False
        filter_warning = True
        uniform = False
        for key in df.__dask_keys__():
            if match:
                dsk[name, i] = key
            else:
                dsk[name, i] = (apply, methods.concat, [[empty, key], 0, join, uniform, filter_warning], kwargs)
            i += 1
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=astyped_dfs)
    return new_dd_object(graph, name, meta, divisions)