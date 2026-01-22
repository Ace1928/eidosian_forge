from statsmodels.compat.platform import PLATFORM_WIN32
import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
def test_constant_integrated_model_error():
    with pytest.raises(ValueError, match='In models with integration'):
        ARIMA(np.ones(100), order=(1, 1, 0), trend='c')
    with pytest.raises(ValueError, match='In models with integration'):
        ARIMA(np.ones(100), order=(1, 0, 0), seasonal_order=(1, 1, 0, 6), trend='c')
    with pytest.raises(ValueError, match='In models with integration'):
        ARIMA(np.ones(100), order=(1, 2, 0), trend='t')
    with pytest.raises(ValueError, match='In models with integration'):
        ARIMA(np.ones(100), order=(1, 1, 0), seasonal_order=(1, 1, 0, 6), trend='t')