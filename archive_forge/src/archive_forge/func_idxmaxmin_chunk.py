from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def idxmaxmin_chunk(x, fn=None, skipna=True, numeric_only=False):
    numeric_only_kwargs = {} if not PANDAS_GE_150 or is_series_like(x) else {'numeric_only': numeric_only}
    minmax = 'max' if fn == 'idxmax' else 'min'
    if len(x) > 0:
        idx = getattr(x, fn)(skipna=skipna, **numeric_only_kwargs)
        value = getattr(x, minmax)(skipna=skipna, **numeric_only_kwargs)
    else:
        idx = value = meta_series_constructor(x)([], dtype='i8')
    if is_series_like(idx):
        return meta_frame_constructor(x)({'idx': idx, 'value': value})
    return meta_frame_constructor(x)({'idx': [idx], 'value': [value]})