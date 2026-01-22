import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_concentrated_initialization():
    mod1 = ExponentialSmoothing(oildata, initialization_method='concentrated')
    mod2 = ExponentialSmoothing(oildata)
    res1 = mod1.filter([0.1])
    res2 = mod2.fit_constrained({'smoothing_level': 0.1}, disp=0)
    res1 = mod1.fit(disp=0)
    res2 = mod2.fit(disp=0)
    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.initial_state, res2.initial_state, rtol=1e-05)