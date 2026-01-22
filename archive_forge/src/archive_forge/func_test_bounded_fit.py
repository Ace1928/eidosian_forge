from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_bounded_fit(oildata):
    beta = [0.99, 0.99]
    model1 = ETSModel(oildata, error='add', trend='add', damped_trend=True, bounds={'smoothing_trend': beta})
    fit1 = model1.fit(disp=False)
    assert fit1.smoothing_trend == 0.99
    model2 = ETSModel(oildata, error='add', trend='add', damped_trend=True)
    with model2.fix_params({'smoothing_trend': 0.99}):
        fit2 = model2.fit(disp=False)
    assert fit2.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit2.params)
    fit2.summary()
    fit3 = model2.fit_constrained({'smoothing_trend': 0.99})
    assert fit3.smoothing_trend == 0.99
    assert_allclose(fit1.params, fit3.params)
    fit3.summary()