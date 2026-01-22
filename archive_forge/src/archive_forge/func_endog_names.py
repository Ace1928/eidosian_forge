from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
@property
def endog_names(self, latex=False):
    """Names of endogenous variables"""
    diff = ''
    if self.k_diff > 0:
        if self.k_diff == 1:
            diff = '\\Delta' if latex else 'D'
        else:
            diff = ('\\Delta^%d' if latex else 'D%d') % self.k_diff
    seasonal_diff = ''
    if self.k_seasonal_diff > 0:
        if self.k_seasonal_diff == 1:
            seasonal_diff = ('\\Delta_%d' if latex else 'DS%d') % self.seasonal_periods
        else:
            seasonal_diff = ('\\Delta_%d^%d' if latex else 'D%dS%d') % (self.k_seasonal_diff, self.seasonal_periods)
    endog_diff = self.simple_differencing
    if endog_diff and self.k_diff > 0 and (self.k_seasonal_diff > 0):
        return ('%s%s %s' if latex else '%s.%s.%s') % (diff, seasonal_diff, self.data.ynames)
    elif endog_diff and self.k_diff > 0:
        return ('%s %s' if latex else '%s.%s') % (diff, self.data.ynames)
    elif endog_diff and self.k_seasonal_diff > 0:
        return ('%s %s' if latex else '%s.%s') % (seasonal_diff, self.data.ynames)
    else:
        return self.data.ynames