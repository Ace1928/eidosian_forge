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
def test_slim_summary(reset_randomstate):
    y = np.random.standard_normal(100)
    x = np.random.standard_normal((100, 1))
    x = x + np.random.standard_normal((100, 5))
    res = OLS(y, x).fit()
    import copy
    summ = copy.deepcopy(res.summary())
    slim_summ = copy.deepcopy(res.summary(slim=True))
    assert len(summ.tables) == 3
    assert len(slim_summ.tables) == 2
    assert summ.tables[0].as_text() != slim_summ.tables[0].as_text()
    assert slim_summ.tables[1].as_text() == summ.tables[1].as_text()