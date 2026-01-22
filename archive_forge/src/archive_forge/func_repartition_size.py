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
def repartition_size(df, size):
    """
    Repartition dataframe so that new partitions have approximately `size` memory usage each
    """
    if isinstance(size, str):
        size = parse_bytes(size)
    size = int(size)
    mem_usages = df.map_partitions(total_mem_usage, deep=True).compute()
    nsplits = 1 + mem_usages // size
    if np.any(nsplits > 1):
        split_name = f'repartition-split-{size}-{tokenize(df)}'
        df = _split_partitions(df, nsplits, split_name)
        split_mem_usages = []
        for n, usage in zip(nsplits, mem_usages):
            split_mem_usages.extend([usage / n] * n)
        mem_usages = pd.Series(split_mem_usages)
    assert np.all(mem_usages <= size)
    new_npartitions = list(map(len, iter_chunks(mem_usages, size)))
    new_partitions_boundaries = np.cumsum(new_npartitions)
    new_name = f'repartition-{size}-{tokenize(df)}'
    return _repartition_from_boundaries(df, new_partitions_boundaries, new_name)