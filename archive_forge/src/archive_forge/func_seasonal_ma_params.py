import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@seasonal_ma_params.setter
def seasonal_ma_params(self, value):
    if np.isscalar(value):
        value = [value] * self.k_seasonal_ma_params
    self._params_split['seasonal_ma_params'] = validate_basic(value, self.k_seasonal_ma_params, title='seasonal MA coefficients')
    self._params = None