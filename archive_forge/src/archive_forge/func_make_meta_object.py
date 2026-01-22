from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
@make_meta_obj.register(meta_object_types)
def make_meta_object(x, index=None):
    """Create an empty pandas object containing the desired metadata.

    Parameters
    ----------
    x : dict, tuple, list, pd.Series, pd.DataFrame, pd.Index, dtype, scalar
        To create a DataFrame, provide a `dict` mapping of `{name: dtype}`, or
        an iterable of `(name, dtype)` tuples. To create a `Series`, provide a
        tuple of `(name, dtype)`. If a pandas object, names, dtypes, and index
        should match the desired output. If a dtype or scalar, a scalar of the
        same dtype is returned.
    index :  pd.Index, optional
        Any pandas index to use in the metadata. If none provided, a
        `RangeIndex` will be used.

    Examples
    --------

    >>> make_meta_object([('a', 'i8'), ('b', 'O')])
    Empty DataFrame
    Columns: [a, b]
    Index: []
    >>> make_meta_object(('a', 'f8'))
    Series([], Name: a, dtype: float64)
    >>> make_meta_object('i8')
    1
    """
    if is_arraylike(x) and x.shape:
        return x[:0]
    if index is not None:
        index = make_meta_dispatch(index)
    if isinstance(x, dict):
        return pd.DataFrame({c: _empty_series(c, d, index=index) for c, d in x.items()}, index=index)
    if isinstance(x, tuple) and len(x) == 2:
        return _empty_series(x[0], x[1], index=index)
    elif isinstance(x, Iterable) and (not isinstance(x, str)):
        if not all((isinstance(i, tuple) and len(i) == 2 for i in x)):
            raise ValueError(f'Expected iterable of tuples of (name, dtype), got {x}')
        return pd.DataFrame({c: _empty_series(c, d, index=index) for c, d in x}, columns=[c for c, d in x], index=index)
    elif not hasattr(x, 'dtype') and x is not None:
        try:
            dtype = np.dtype(x)
            return _scalar_from_dtype(dtype)
        except Exception:
            pass
    if is_scalar(x):
        return _nonempty_scalar(x)
    raise TypeError(f"Don't know how to create metadata from {x}")