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
def set_index_columns(meta, index, columns, auto_index_allowed):
    """Handle index/column arguments, and modify `meta`
    Used in read_parquet.
    """
    ignore_index_column_intersection = False
    if columns is None:
        ignore_index_column_intersection = True
        _index = index or []
        columns = [c for c in meta.columns if c not in (None, NONE_LABEL) or c in _index]
    if not set(columns).issubset(set(meta.columns)):
        raise ValueError('The following columns were not found in the dataset %s\nThe following columns were found %s' % (set(columns) - set(meta.columns), meta.columns))
    if index:
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]
        if ignore_index_column_intersection:
            columns = [col for col in columns if col not in index]
        if set(index).intersection(columns):
            if auto_index_allowed:
                raise ValueError('Specified index and column arguments must not intersect (set index=False or remove the detected index from columns).\nindex: {} | column: {}'.format(index, columns))
            else:
                raise ValueError('Specified index and column arguments must not intersect.\nindex: {} | column: {}'.format(index, columns))
    return (meta[list(columns)], index, columns)