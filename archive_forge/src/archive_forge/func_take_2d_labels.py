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
def take_2d_labels(self, index, columns):
    """
        Take the given labels.

        Parameters
        ----------
        index : slice, scalar, list-like, or BaseQueryCompiler
            Labels of rows to grab.
        columns : slice, scalar, list-like, or BaseQueryCompiler
            Labels of columns to grab.

        Returns
        -------
        BaseQueryCompiler
            Subset of this QueryCompiler.
        """
    row_lookup, col_lookup = self.get_positions_from_labels(index, columns)
    if isinstance(row_lookup, slice):
        ErrorMessage.catch_bugs_and_request_email(failure_condition=row_lookup != slice(None), extra_log=f'Only None-slices are acceptable as a slice argument in masking, got: {row_lookup}')
        row_lookup = None
    if isinstance(col_lookup, slice):
        ErrorMessage.catch_bugs_and_request_email(failure_condition=col_lookup != slice(None), extra_log=f'Only None-slices are acceptable as a slice argument in masking, got: {col_lookup}')
        col_lookup = None
    return self.take_2d_positional(row_lookup, col_lookup)