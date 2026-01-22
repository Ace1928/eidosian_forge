import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_sign_test():
    x = [7.8, 6.6, 6.5, 7.4, 7.3, 7.0, 6.4, 7.1, 6.7, 7.6, 6.8]
    M, p = sign_test(x, mu0=6.5)
    assert_almost_equal(p, 0.02148, 5)
    assert_equal(M, 4)