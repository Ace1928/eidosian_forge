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
def check_cloned(mod, endog, exog=None):
    mod_c = mod.clone(endog, exog=exog)
    assert_allclose(mod.nobs, mod_c.nobs)
    assert_(mod._index.equals(mod_c._index))
    assert_equal(mod.k_params, mod_c.k_params)
    assert_allclose(mod.start_params, mod_c.start_params)
    p = mod.start_params
    assert_allclose(mod.loglike(p), mod_c.loglike(p))
    assert_allclose(mod.concentrate_scale, mod_c.concentrate_scale)