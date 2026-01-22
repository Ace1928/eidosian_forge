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
def merge_chunk(lhs, *args, result_meta, **kwargs):
    rhs, *args = args
    left_index = kwargs.get('left_index', False)
    right_index = kwargs.get('right_index', False)
    empty_index_dtype = result_meta.index.dtype
    categorical_columns = result_meta.select_dtypes(include='category').columns
    if categorical_columns is not None:
        for col in categorical_columns:
            left = None
            right = None
            if col in lhs:
                left = lhs[col]
            elif col == kwargs.get('right_on', None) and left_index:
                if isinstance(lhs.index.dtype, pd.CategoricalDtype):
                    left = lhs.index
            if col in rhs:
                right = rhs[col]
            elif col == kwargs.get('left_on', None) and right_index:
                if isinstance(rhs.index.dtype, pd.CategoricalDtype):
                    right = rhs.index
            dtype = 'category'
            if left is not None and right is not None:
                dtype = methods.union_categoricals([left.astype('category'), right.astype('category')]).dtype
            if left is not None:
                if isinstance(left, pd.Index):
                    lhs.index = left.astype(dtype)
                else:
                    lhs = lhs.assign(**{col: left.astype(dtype)})
            if right is not None:
                if isinstance(right, pd.Index):
                    rhs.index = right.astype(dtype)
                else:
                    rhs = rhs.assign(**{col: right.astype(dtype)})
    if len(args) and args[0] == 'leftsemi' or kwargs.get('how', None) == 'leftsemi':
        if isinstance(rhs, (pd.DataFrame, pd.Series)):
            rhs = rhs.drop_duplicates()
            if len(args):
                args[0] = 'inner'
            else:
                kwargs['how'] = 'inner'
    out = lhs.merge(rhs, *args, **kwargs)
    if len(lhs) == 0:
        out = out[result_meta.columns]
    if len(out) == 0 and empty_index_dtype is not None:
        out.index = out.index.astype(empty_index_dtype)
    return out