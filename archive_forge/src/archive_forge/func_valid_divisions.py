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
def valid_divisions(divisions):
    """Are the provided divisions valid?

    Examples
    --------
    >>> valid_divisions([1, 2, 3])
    True
    >>> valid_divisions([3, 2, 1])
    False
    >>> valid_divisions([1, 1, 1])
    False
    >>> valid_divisions([0, 1, 1])
    True
    >>> valid_divisions((1, 2, 3))
    True
    >>> valid_divisions(123)
    False
    >>> valid_divisions([0, float('nan'), 1])
    False
    """
    if not isinstance(divisions, (tuple, list)):
        return False
    if isinstance(divisions, tuple):
        divisions = list(divisions)
    if pd.isnull(divisions).any():
        return False
    for i, x in enumerate(divisions[:-2]):
        if x >= divisions[i + 1]:
            return False
        if isinstance(x, Number) and math.isnan(x):
            return False
    for x in divisions[-2:]:
        if isinstance(x, Number) and math.isnan(x):
            return False
    if divisions[-2] > divisions[-1]:
        return False
    return True