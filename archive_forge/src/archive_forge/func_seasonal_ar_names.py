import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def seasonal_ar_names(self):
    """(list of str) Names of seasonal autoregressive parameters."""
    s = self.seasonal_periods
    return ['ar.S.L%d' % (i * s) for i in self.seasonal_ar_lags]