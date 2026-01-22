from __future__ import annotations
import contextlib
import math
import warnings
from typing import Literal
import pandas as pd
import tlz as toolz
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
import dask
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import DataFrameIOFunction, _is_local_fs
from dask.dataframe.methods import concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import apply, import_required, natural_sort_key, parse_bytes
def read_parquet_part(fs, engine, meta, part, columns, index, kwargs):
    """Read a part of a parquet dataset

    This function is used by `read_parquet`."""
    if isinstance(part, list):
        if len(part) == 1 or part[0][1] or (not check_multi_support(engine)):
            func = engine.read_partition
            dfs = [func(fs, rg, columns.copy(), index, **toolz.merge(kwargs, kw)) for rg, kw in part]
            df = concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
        else:
            df = engine.read_partition(fs, [p[0] for p in part], columns.copy(), index, **kwargs)
    else:
        rg, part_kwargs = part
        df = engine.read_partition(fs, rg, columns, index, **toolz.merge(kwargs, part_kwargs))
    if meta.columns.name:
        df.columns.name = meta.columns.name
    columns = columns or []
    index = index or []
    df = df[[c for c in columns if c not in index]]
    if index == [NONE_LABEL]:
        df.index.name = None
    return df