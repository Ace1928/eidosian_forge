from __future__ import annotations
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Number
from typing import TypeVar, overload
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
import dask
from dask.base import get_scheduler, is_dask_collection
from dask.core import get_deps
from dask.dataframe import (  # noqa: F401 register pandas extension types
from dask.dataframe._compat import PANDAS_GE_150, tm  # noqa: F401
from dask.dataframe.dispatch import (  # noqa : F401
from dask.dataframe.extensions import make_scalar
from dask.typing import NoDefault, no_default
from dask.utils import (
def clear_known_categories(x, cols=None, index=True, dtype_backend=None):
    """Set categories to be unknown.

    Parameters
    ----------
    x : DataFrame, Series, Index
    cols : iterable, optional
        If x is a DataFrame, set only categoricals in these columns to unknown.
        By default, all categorical columns are set to unknown categoricals
    index : bool, optional
        If True and x is a Series or DataFrame, set the clear known categories
        in the index as well.
    dtype_backend : string, optional
        If set to PyArrow, the categorical dtype is implemented as a PyArrow
        dictionary
    """
    if dtype_backend == 'pyarrow':
        return x
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.copy()
        if isinstance(x, pd.DataFrame):
            mask = x.dtypes == 'category'
            if cols is None:
                cols = mask[mask].index
            elif not mask.loc[cols].all():
                raise ValueError('Not all columns are categoricals')
            for c in cols:
                x[c] = x[c].cat.set_categories([UNKNOWN_CATEGORIES])
        elif isinstance(x, pd.Series):
            if isinstance(x.dtype, pd.CategoricalDtype):
                x = x.cat.set_categories([UNKNOWN_CATEGORIES])
        if index and isinstance(x.index, pd.CategoricalIndex):
            x.index = x.index.set_categories([UNKNOWN_CATEGORIES])
    elif isinstance(x, pd.CategoricalIndex):
        x = x.set_categories([UNKNOWN_CATEGORIES])
    return x