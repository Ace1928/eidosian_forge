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
def single_partition_join(left, right, **kwargs):
    meta = left._meta_nonempty.merge(right._meta_nonempty, **kwargs)
    use_left = kwargs.get('right_index') or right._contains_index_name(kwargs.get('right_on'))
    use_right = kwargs.get('left_index') or left._contains_index_name(kwargs.get('left_on'))
    if len(meta) == 0:
        if use_left:
            meta.index = meta.index.astype(left.index.dtype)
        elif use_right:
            meta.index = meta.index.astype(right.index.dtype)
        else:
            meta.index = meta.index.astype('int64')
    kwargs['result_meta'] = meta
    if right.npartitions == 1 and kwargs['how'] in allowed_left:
        if use_left:
            divisions = left.divisions
        elif use_right and len(right.divisions) == len(left.divisions):
            divisions = right.divisions
        else:
            divisions = [None for _ in left.divisions]
    elif left.npartitions == 1 and kwargs['how'] in allowed_right:
        if use_right:
            divisions = right.divisions
        elif use_left and len(left.divisions) == len(right.divisions):
            divisions = left.divisions
        else:
            divisions = [None for _ in right.divisions]
    else:
        raise NotImplementedError('single_partition_join has no fallback for invalid calls')
    joined = map_partitions(merge_chunk, left, right, meta=meta, enforce_metadata=False, transform_divisions=False, align_dataframes=False, **kwargs)
    joined.divisions = tuple(divisions)
    return joined