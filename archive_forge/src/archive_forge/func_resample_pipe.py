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
@doc_utils.add_refer_to('Resampler.pipe')
def resample_pipe(self, resample_kwargs, func, *args, **kwargs):
    """
        Resample time-series data and apply aggregation on it.

        Group data into intervals by time-series row/column with
        a specified frequency, build equivalent ``pandas.Resampler`` object
        and apply passed function to it.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        func : callable(pandas.Resampler) -> object or tuple(callable, str)
        *args : iterable
            Positional arguments to pass to function.
        **kwargs : dict
            Keyword arguments to pass to function.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the result of passed function.
        """
    return ResampleDefault.register(pandas.core.resample.Resampler.pipe)(self, resample_kwargs, func, *args, **kwargs)