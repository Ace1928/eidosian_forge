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
@divisions.setter
def divisions(self, value):
    if not isinstance(value, tuple):
        raise TypeError('divisions must be a tuple')
    if hasattr(self, '_divisions') and len(value) != len(self._divisions):
        n = len(self._divisions)
        raise ValueError(f'This dataframe has npartitions={n - 1}, divisions should be a tuple of length={n}, got {len(value)}')
    if None in value:
        if any((v is not None for v in value)):
            raise ValueError('divisions may not contain a mix of None and non-None values')
    else:
        index_dtype = getattr(self._meta, 'index', self._meta).dtype
        if not (isinstance(index_dtype, pd.CategoricalDtype) and index_dtype.ordered):
            if value != tuple(sorted(value)):
                raise ValueError('divisions must be sorted')
    self._divisions = value