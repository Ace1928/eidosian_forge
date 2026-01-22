from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def attach_columns(self, result):
    if result.ndim <= 1:
        return Series(result, index=self.param_names)
    else:
        return DataFrame(result, index=self.param_names)