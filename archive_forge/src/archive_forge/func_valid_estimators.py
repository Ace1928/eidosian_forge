import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def valid_estimators(self):
    """
        (list of str) Estimators that could be used with specification.

        Note: does not consider the presense of `exog` in determining valid
        estimators. If there are exogenous variables, then feasible Generalized
        Least Squares should be used through the `gls` estimator, and the
        `valid_estimators` are the estimators that could be passed as the
        `arma_estimator` argument to `gls`.
        """
    estimators = {'yule_walker', 'burg', 'innovations', 'hannan_rissanen', 'innovations_mle', 'statespace'}
    has_ar = self.max_ar_order != 0
    has_ma = self.max_ma_order != 0
    has_seasonal = self.seasonal_periods != 0
    if self._has_missing:
        estimators.intersection_update(['statespace'])
    if self.enforce_stationarity and self.max_ar_order > 0 or (self.enforce_invertibility and self.max_ma_order > 0):
        estimators.intersection_update(['innovations_mle', 'statespace'])
    if has_ar or not self.is_ma_consecutive or has_seasonal:
        estimators.discard('innovations')
    if has_ma or not self.is_ar_consecutive or has_seasonal:
        estimators.discard('yule_walker')
        estimators.discard('burg')
    if has_seasonal:
        estimators.discard('hannan_rissanen')
    if self.enforce_stationarity is False or self.concentrate_scale:
        estimators.discard('innovations_mle')
    return estimators