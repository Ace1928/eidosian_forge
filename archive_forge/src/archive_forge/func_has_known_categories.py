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
def has_known_categories(x):
    """Returns whether the categories in `x` are known.

    Parameters
    ----------
    x : Series or CategoricalIndex
    """
    x = getattr(x, '_meta', x)
    if is_series_like(x):
        return UNKNOWN_CATEGORIES not in x.cat.categories
    elif is_index_like(x) and hasattr(x, 'categories'):
        return UNKNOWN_CATEGORIES not in x.categories
    raise TypeError('Expected Series or CategoricalIndex')