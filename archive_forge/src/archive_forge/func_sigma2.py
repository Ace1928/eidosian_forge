import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@sigma2.setter
def sigma2(self, params):
    length = int(not self.spec.concentrate_scale)
    self._params_split['sigma2'] = validate_basic(params, length, title='sigma2').item()
    self._params = None