import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
def resample_asfreq(self, resample_kwargs, fill_value):
    """
        Resample time-series data and get the values at the new frequency.

        Group data into intervals by time-series row/column with
        a specified frequency and get values at the new frequency.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        fill_value : scalar

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing values at the specified frequency.
        """
    return ResampleDefault.register(pandas.core.resample.Resampler.asfreq)(self, resample_kwargs, fill_value)