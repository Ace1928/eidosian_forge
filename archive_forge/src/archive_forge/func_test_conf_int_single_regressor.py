from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_conf_int_single_regressor():
    y = pd.Series(np.random.randn(10))
    x = pd.Series(np.ones(10))
    res = OLS(y, x).fit()
    conf_int = res.conf_int()
    np.testing.assert_equal(conf_int.shape, (1, 2))
    np.testing.assert_(isinstance(conf_int, pd.DataFrame))