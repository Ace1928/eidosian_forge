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
@doc_utils.add_refer_to('DataFrame.query')
def rowwise_query(self, expr, **kwargs):
    """
        Query columns of the QueryCompiler with a boolean expression row-wise.

        Parameters
        ----------
        expr : str
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the rows where the boolean expression is satisfied.
        """
    raise NotImplementedError('Row-wise queries execution is not implemented for the selected backend.')