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
def test_predict_vs_R(setup_model):
    model, params, results_R = setup_model
    fit = fit_austourists_with_R_params(model, results_R, set_state=True)
    n = fit.nobs
    prediction = fit.predict(end=n + 3, dynamic=n)
    yhat_R = results_R['fitted']
    assert_allclose(prediction[:n], yhat_R, rtol=1e-05, atol=1e-05)
    forecast_R = results_R['forecast']
    assert_allclose(prediction[n:], forecast_R, rtol=0.001, atol=0.0001)