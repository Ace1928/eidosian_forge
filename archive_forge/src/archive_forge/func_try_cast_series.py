import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
def try_cast_series(df):
    """Cast one-column frame to Series."""
    if isinstance(df, pandas.DataFrame):
        df = df.squeeze(axis=1)
    if not isinstance(df, pandas.Series):
        return df
    if df.name == MODIN_UNNAMED_SERIES_LABEL:
        df.name = None
    return df