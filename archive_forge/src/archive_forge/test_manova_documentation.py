import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant

    assert_raises_regex(ValueError,
                        ('Transform matrix M should have the same number of '
                         'rows as the number of columns of endog! 2 != 3'),
                        mod.mv_test, hypothesis)
    