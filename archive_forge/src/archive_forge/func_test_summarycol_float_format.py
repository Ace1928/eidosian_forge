import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_summarycol_float_format(self):
    desired = '\n==========================\n                y I   y II\n--------------------------\nconst          7.7   12.4 \n               (1.1) (3.2)\nx1             -0.7  -1.6 \n               (0.2) (0.7)\nR-squared      0.8   0.6  \nR-squared Adj. 0.7   0.5  \n==========================\nStandard errors in\nparentheses.\n'
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y1 = [6, 4, 2, 7, 4]
    y2 = [8, 5, 0, 12, 4]
    reg1 = OLS(y1, x).fit()
    reg2 = OLS(y2, x).fit()
    actual = summary_col([reg1, reg2], float_format='%0.1f').as_text()
    actual = '%s\n' % actual
    starred = summary_col([reg1, reg2], stars=True, float_format='%0.1f')
    assert '7.7***' in str(starred)
    assert '12.4**' in str(starred)
    assert '12.4***' not in str(starred)
    assert_equal(actual, desired)