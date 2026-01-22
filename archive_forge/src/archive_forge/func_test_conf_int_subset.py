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
def test_conf_int_subset(self):
    if len(self.res1.params) > 1:
        with pytest.warns(FutureWarning, match='cols is'):
            ci1 = self.res1.conf_int(cols=(1, 2))
        ci2 = self.res1.conf_int()[1:3]
        assert_almost_equal(ci1, ci2, self.decimal_conf_int_subset)
    else:
        pass