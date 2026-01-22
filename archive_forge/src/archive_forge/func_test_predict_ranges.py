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
def test_predict_ranges(austourists_model_fit):
    fit = austourists_model_fit
    pred = fit.predict(start=0, end=10)
    assert len(pred) == 11
    pred = fit.predict(start=10, end=20)
    assert len(pred) == 11
    pred = fit.predict(start=10, dynamic=10, end=30)
    assert len(pred) == 21
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71
    pred = fit.predict(start=0, dynamic=True, end=70)
    assert len(pred) == 71
    pred = fit.predict(start=80, end=84)
    assert len(pred) == 5