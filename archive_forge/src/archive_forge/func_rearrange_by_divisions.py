from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
@_deprecated_kwarg('shuffle', 'shuffle_method')
def rearrange_by_divisions(df, column, divisions, max_branch=None, shuffle_method=None, ascending=True, na_position='last', duplicates=True):
    """Shuffle dataframe so that column separates along divisions"""
    divisions = df._meta._constructor_sliced(divisions)
    if not duplicates:
        divisions = divisions.drop_duplicates()
    meta = df._meta._constructor_sliced([0])
    meta.index = df._meta_nonempty.index[:1]
    partitions = df[column].map_partitions(set_partitions_pre, divisions=divisions, ascending=ascending, na_position=na_position, meta=meta)
    df2 = df.assign(_partitions=partitions)
    df3 = rearrange_by_column(df2, '_partitions', max_branch=max_branch, npartitions=len(divisions) - 1, shuffle_method=shuffle_method)
    del df3['_partitions']
    return df3