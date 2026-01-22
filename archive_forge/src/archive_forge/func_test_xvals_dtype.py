import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_xvals_dtype(reset_randomstate):
    y = [0] * 10 + [1] * 10
    x = np.arange(20)
    results_xvals = lowess(y, x, frac=0.4, xvals=x[:5])
    assert_allclose(results_xvals, np.zeros(5), atol=1e-12)