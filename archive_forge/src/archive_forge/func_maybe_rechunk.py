from __future__ import annotations
import typing
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.dtypes import (
import pandas as pd
def maybe_rechunk(series: pd.Series, *, allow_copy: bool) -> pd.Series | None:
    """
    Rechunk a multi-chunk pyarrow array into a single-chunk array, if necessary.

    - Returns `None` if the input series is not backed by a multi-chunk pyarrow array
      (and so doesn't need rechunking)
    - Returns a single-chunk-backed-Series if the input is backed by a multi-chunk
      pyarrow array and `allow_copy` is `True`.
    - Raises a `RuntimeError` if `allow_copy` is `False` and input is a
      based by a multi-chunk pyarrow array.
    """
    if not isinstance(series.dtype, pd.ArrowDtype):
        return None
    chunked_array = series.array._pa_array
    if len(chunked_array.chunks) == 1:
        return None
    if not allow_copy:
        raise RuntimeError('Found multi-chunk pyarrow array, but `allow_copy` is False. Please rechunk the array before calling this function, or set `allow_copy=True`.')
    arr = chunked_array.combine_chunks()
    return pd.Series(arr, dtype=series.dtype, name=series.name, index=series.index)