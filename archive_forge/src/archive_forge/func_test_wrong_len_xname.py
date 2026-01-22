import numpy as np  # noqa: F401
import pytest
from numpy.testing import assert_equal
from statsmodels.datasets import macrodata
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_wrong_len_xname(reset_randomstate):
    y = np.random.randn(100)
    x = np.random.randn(100, 2)
    res = OLS(y, x).fit()
    with pytest.raises(ValueError):
        res.summary(xname=['x1'])
    with pytest.raises(ValueError):
        res.summary(xname=['x1', 'x2', 'x3'])