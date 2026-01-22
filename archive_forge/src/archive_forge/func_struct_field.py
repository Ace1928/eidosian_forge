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
@doc_utils.add_refer_to('Series.struct.field')
def struct_field(self, name_or_index):
    """
        Extract a child field of a struct as a Series.

        Returns
        -------
        BaseQueryCompiler
        """
    return StructDefault.register(pandas.Series.struct.field)(self, name_or_index=name_or_index)