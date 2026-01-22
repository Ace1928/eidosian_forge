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
def series_map(base_series, map_series):
    npartitions = base_series.npartitions
    split_out = map_series.npartitions
    dsk = {}
    base_token_key = tokenize(base_series, split_out)
    base_split_prefix = f'base-split-{base_token_key}'
    base_shard_prefix = f'base-shard-{base_token_key}'
    for i, key in enumerate(base_series.__dask_keys__()):
        dsk[base_split_prefix, i] = (hash_shard, key, split_out)
        for j in range(split_out):
            dsk[base_shard_prefix, 0, i, j] = (getitem, (base_split_prefix, i), j)
    map_token_key = tokenize(map_series)
    map_split_prefix = f'map-split-{map_token_key}'
    map_shard_prefix = f'map-shard-{map_token_key}'
    for i, key in enumerate(map_series.__dask_keys__()):
        dsk[map_split_prefix, i] = (hash_shard, key, split_out, split_out_on_index, None)
        for j in range(split_out):
            dsk[map_shard_prefix, 0, i, j] = (getitem, (map_split_prefix, i), j)
    token_key = tokenize(base_series, map_series)
    map_prefix = f'map-series-{token_key}'
    for i in range(npartitions):
        for j in range(split_out):
            dsk[map_prefix, i, j] = (mapseries, (base_shard_prefix, 0, i, j), (_concat, [(map_shard_prefix, 0, k, j) for k in range(split_out)]))
    final_prefix = f'map-series-combine-{token_key}'
    for i, key in enumerate(base_series.index.__dask_keys__()):
        dsk[final_prefix, i] = (mapseries_combine, key, (_concat, [(map_prefix, i, j) for j in range(split_out)]))
    meta = map_series._meta.copy()
    meta.index = base_series._meta.index
    meta = make_meta(meta)
    dependencies = [base_series, map_series, base_series.index]
    graph = HighLevelGraph.from_collections(final_prefix, dsk, dependencies=dependencies)
    divisions = list(base_series.divisions)
    return new_dd_object(graph, final_prefix, meta, divisions)