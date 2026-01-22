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
@doc_utils.doc_window_method(window_cls_name='Rolling', result='variance', refer_to='var', params='\n        ddof : int, default: 1\n        *args : iterable\n        **kwargs : dict')
def rolling_var(self, fold_axis, rolling_kwargs, ddof=1, *args, **kwargs):
    return RollingDefault.register(pandas.core.window.rolling.Rolling.var)(self, rolling_kwargs, ddof, *args, **kwargs)