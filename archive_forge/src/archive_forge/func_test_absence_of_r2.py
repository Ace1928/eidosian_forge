import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_absence_of_r2(self):
    table = summary_col(results=self.mod, include_r2=False)
    assert 'R-squared' not in str(table)
    assert 'R-squared Adj.' not in str(table)