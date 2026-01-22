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
def default_to_pandas(self, pandas_op, *args, **kwargs):
    """
        Do fallback to pandas for the passed function.

        Parameters
        ----------
        pandas_op : callable(pandas.DataFrame) -> object
            Function to apply to the casted to pandas frame.
        *args : iterable
            Positional arguments to pass to `pandas_op`.
        **kwargs : dict
            Key-value arguments to pass to `pandas_op`.

        Returns
        -------
        BaseQueryCompiler
            The result of the `pandas_op`, converted back to ``BaseQueryCompiler``.
        """
    op_name = getattr(pandas_op, '__name__', str(pandas_op))
    ErrorMessage.default_to_pandas(op_name)
    args = try_cast_to_pandas(args)
    kwargs = try_cast_to_pandas(kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        result = pandas_op(try_cast_to_pandas(self), *args, **kwargs)
    if isinstance(result, (tuple, list)):
        if 'Series.tolist' in pandas_op.__name__:
            return result
        return [self.__wrap_in_qc(obj) for obj in result]
    return self.__wrap_in_qc(result)