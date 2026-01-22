from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def safe_cast_to_index(array: Any) -> pd.Index:
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.variable import Variable
    if isinstance(array, pd.Index):
        index = array
    elif isinstance(array, (DataArray, Variable)):
        index = array._to_index()
    elif isinstance(array, Index):
        index = array.to_pandas_index()
    elif isinstance(array, PandasIndexingAdapter):
        index = array.array
    else:
        kwargs: dict[str, str] = {}
        if hasattr(array, 'dtype'):
            if array.dtype.kind == 'O':
                kwargs['dtype'] = 'object'
            elif array.dtype == 'float16':
                emit_user_level_warning('`pandas.Index` does not support the `float16` dtype. Casting to `float64` for you, but in the future please manually cast to either `float32` and `float64`.', category=DeprecationWarning)
                kwargs['dtype'] = 'float64'
        index = pd.Index(np.asarray(array), **kwargs)
    return _maybe_cast_to_cftimeindex(index)