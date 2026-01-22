import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pytest
from statsmodels.stats.robust_compare import (
import statsmodels.stats.oneway as smo
from statsmodels.tools.testing import Holder
from scipy.stats import trim1
def t_est_trim1(self):
    a = np.arange(11)
    assert_equal(trim1(a, 0.1), np.arange(10))
    assert_equal(trim1(a, 0.2), np.arange(9))
    assert_equal(trim1(a, 0.2, tail='left'), np.arange(2, 11))
    assert_equal(trim1(a, 3 / 11.0, tail='left'), np.arange(3, 11))