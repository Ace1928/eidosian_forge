from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
@cache_writable()
def ynames(self):
    endog = self.orig_endog
    ynames = self._get_names(endog)
    if not ynames:
        ynames = _make_endog_names(self.endog)
    if len(ynames) == 1:
        return ynames[0]
    else:
        return list(ynames)