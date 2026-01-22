from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def transform_series(series: pd.Series[Any] | pd.Index[Any] | pd.api.extensions.ExtensionArray) -> npt.NDArray[Any]:
    """ Transforms a Pandas series into serialized form

    Args:
        series (pd.Series) : the Pandas series to transform

    Returns:
        ndarray

    """
    import pandas as pd
    if isinstance(series, pd.PeriodIndex):
        vals = series.to_timestamp().values
    else:
        vals = series.to_numpy()
    return vals