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
@doc_utils.doc_window_method(window_cls_name='Expanding', result='median', refer_to='median', win_type='expanding window', params='\n        numeric_only : bool, default: False\n        engine : Optional[str], default: None\n        engine_kwargs : Optional[dict], default: None\n        **kwargs : dict')
def expanding_median(self, fold_axis, expanding_args, numeric_only=False, engine=None, engine_kwargs=None, **kwargs):
    return ExpandingDefault.register(pandas.core.window.expanding.Expanding.median)(self, expanding_args, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs, **kwargs)