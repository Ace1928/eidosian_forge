import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_summarycol_drop_omitted(self):
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
    y1 = [6, 4, 2, 7, 4]
    y2 = [8, 5, 0, 12, 4]
    reg1 = OLS(y1, x).fit()
    reg2 = OLS(y2, x2).fit()
    actual = summary_col([reg1, reg2], regressor_order=['const', 'x1'], drop_omitted=True)
    assert 'x2' not in str(actual)
    actual = summary_col([reg1, reg2], regressor_order=['x1'], drop_omitted=False)
    assert 'const' in str(actual)
    assert 'x2' in str(actual)