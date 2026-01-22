import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@seasonal_ma_poly.setter
def seasonal_ma_poly(self, value):
    s = self.spec.seasonal_periods
    if isinstance(value, Polynomial):
        value = value.coef
    value = validate_basic(value, 1 + s * self.spec.max_seasonal_ma_order, title='seasonal MA polynomial')
    if value[0] != 1:
        raise ValueError('Polynomial constant must be equal to 1.')
    seasonal_ma_params = []
    for i in range(1, self.spec.max_seasonal_ma_order + 1):
        if i in self.spec.seasonal_ma_lags:
            seasonal_ma_params.append(value[s * i])
        elif value[s * i] != 0:
            raise ValueError('MA polynomial includes non-zero values for lags that are excluded in the specification.')
    self.seasonal_ma_params = seasonal_ma_params