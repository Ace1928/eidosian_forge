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
@doc_utils.doc_window_method(window_cls_name='Rolling', result='rank', refer_to='rank', params="\n        method : {'average', 'min', 'max'}, default: 'average'\n        ascending : bool, default: True\n        pct : bool, default: False\n        numeric_only : bool, default: False\n        *args : iterable\n        **kwargs : dict")
def rolling_rank(self, fold_axis, rolling_kwargs, method='average', ascending=True, pct=False, numeric_only=False, *args, **kwargs):
    return RollingDefault.register(pandas.core.window.rolling.Rolling.rank)(self, rolling_kwargs, *args, method=method, ascending=ascending, pct=pct, numeric_only=numeric_only, **kwargs)