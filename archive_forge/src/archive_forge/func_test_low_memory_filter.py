import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
def test_low_memory_filter():
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    mod.ssm.set_conserve_memory(MEMORY_NO_GAIN)
    res = mod.filter([0.5], low_memory=True)
    assert_equal(res.filter_results.conserve_memory, MEMORY_CONSERVE)
    assert_(res.llf_obs is None)
    assert_equal(mod.ssm.conserve_memory, MEMORY_NO_GAIN)