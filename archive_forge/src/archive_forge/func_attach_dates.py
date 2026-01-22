from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def attach_dates(self, result):
    squeezed = result.squeeze()
    k_endog = np.array(self.ynames, ndmin=1).shape[0]
    if k_endog > 1 and squeezed.shape == (k_endog,):
        squeezed = np.asarray(squeezed)[None, :]
    if squeezed.ndim < 2:
        return Series(squeezed, index=self.predict_dates)
    else:
        return DataFrame(np.asarray(result), index=self.predict_dates, columns=self.ynames)