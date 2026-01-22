from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
def test_exog_prediction(ar2):
    gen = np.random.RandomState(20210623)
    exog = pd.DataFrame(gen.standard_normal((ar2.shape[0], 2)), columns=['x1', 'x2'], index=ar2.index)
    mod = AutoReg(ar2, 2, trend='c', exog=exog)
    res = mod.fit()
    pred_base = res.predict()
    pred_repl = res.predict(exog=exog)
    assert_allclose(pred_base, pred_repl)
    dyn_base = res.predict(dynamic=25)
    dyn_repl = res.predict(dynamic=25, exog=exog)
    assert_allclose(dyn_base, dyn_repl)