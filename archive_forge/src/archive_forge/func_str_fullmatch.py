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
@doc_utils.doc_str_method(refer_to='fullmatch', params='\n        pat : str\n        case : bool, default: True\n        flags : int, default: 0\n        na : object, default: None')
def str_fullmatch(self, pat, case=True, flags=0, na=None):
    return StrDefault.register(pandas.Series.str.fullmatch)(self, pat, case, flags, na)