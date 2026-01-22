import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant
def test_manova_test_input_validation():
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    hypothesis = [('test', np.array([[1, 1, 1]]), None)]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1]]), None)]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    "\n    assert_raises_regex(ValueError,\n                        ('Contrast matrix L should have the same number of '\n                         'columns as exog! 2 != 3'),\n                        mod.mv_test, hypothesis)\n    "
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1], [1]]))]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1]]))]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    "\n    assert_raises_regex(ValueError,\n                        ('Transform matrix M should have the same number of '\n                         'rows as the number of columns of endog! 2 != 3'),\n                        mod.mv_test, hypothesis)\n    "