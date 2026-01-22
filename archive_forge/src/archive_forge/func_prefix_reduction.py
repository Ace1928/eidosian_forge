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
def prefix_reduction(f, ddf, identity, **kwargs):
    """Computes the prefix sums of f on df

    If df has partitions [P1, P2, ..., Pn], then returns the DataFrame with
    partitions [f(identity, P1),
                f(f(identity, P1), P2),
                f(f(f(identity, P1), P2), P3),
                ...]

    Parameters
    ----------
    f : callable
        an associative function f
    ddf : dd.DataFrame
    identity : pd.DataFrame
        an identity element of f, that is f(identity, df) = f(df, identity) = df
    """
    dsk = dict()
    name = 'prefix_reduction-' + tokenize(f, ddf, identity, **kwargs)
    meta = ddf._meta
    n = len(ddf.divisions) - 1
    divisions = [None] * (n + 1)
    N = 1
    while N < n:
        N *= 2
    for i in range(n):
        dsk[name, i, 1, 0] = (apply, f, [(ddf._name, i), identity], kwargs)
    for i in range(n, N):
        dsk[name, i, 1, 0] = identity
    d = 1
    while d < N:
        for i in range(0, N, 2 * d):
            dsk[name, i + 2 * d - 1, 2 * d, 0] = (apply, f, [(name, i + d - 1, d, 0), (name, i + 2 * d - 1, d, 0)], kwargs)
        d *= 2
    dsk[name, N - 1, N, 1] = identity
    while d > 1:
        d //= 2
        for i in range(0, N, 2 * d):
            dsk[name, i + d - 1, d, 1] = (name, i + 2 * d - 1, 2 * d, 1)
            dsk[name, i + 2 * d - 1, d, 1] = (apply, f, [(name, i + 2 * d - 1, 2 * d, 1), (name, i + d - 1, d, 0)], kwargs)
    for i in range(n):
        dsk[name, i] = (apply, f, [(name, i, 1, 1), identity], kwargs)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
    return new_dd_object(graph, name, meta, divisions)