import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_kernel_cumincidence1():
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(0, 10, size=n)
    status = np.random.randint(0, 3, size=n)
    result1 = CumIncidenceRight(time, status)
    for dimred in (False, True):
        result2 = CumIncidenceRight(time, status, exog=x, bw_factor=10000, dimred=dimred)
        assert_allclose(result1.times, result2.times)
        for k in (0, 1):
            assert_allclose(result1.cinc[k], result2.cinc[k], rtol=1e-05)