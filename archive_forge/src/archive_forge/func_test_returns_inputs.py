import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_returns_inputs():
    y = [0] * 10 + [1] * 10
    x = np.arange(20)
    result = lowess(y, x, frac=0.4)
    assert_almost_equal(result, np.column_stack((x, y)))