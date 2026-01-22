import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_sarimax_starting_values_few_obsevations_long_ma(reset_randomstate):
    y = np.random.standard_normal(9)
    y = [3066.3, 3260.2, 3573.7, 3423.6, 3598.5, 3802.8, 3353.4, 4026.1, 4684.0, 4099.1, 3883.1, 3801.5, 3104.0, 3574.0, 3397.2, 3092.9, 3083.8, 3106.7, 2939.6]
    sarimax_model = sarimax.SARIMAX(endog=y, order=(0, 1, 5), trend='n').fit(disp=False)
    assert np.all(np.isfinite(sarimax_model.predict(start=len(y), end=len(y) + 11)))