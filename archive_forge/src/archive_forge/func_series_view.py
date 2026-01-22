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
@doc_utils.add_one_column_warning
def series_view(self, **kwargs):
    """
        Reinterpret underlying data with new dtype.

        Parameters
        ----------
        dtype : dtype
            Data type to reinterpret underlying data with.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler of the same data in memory, with reinterpreted values.

        Notes
        -----
            - Be aware, that if this method do fallback to pandas, then newly created
              QueryCompiler will be the copy of the original data.
            - Please refer to ``modin.pandas.Series.view`` for more information
              about parameters and output format.
        """
    return SeriesDefault.register(pandas.Series.view)(self, **kwargs)