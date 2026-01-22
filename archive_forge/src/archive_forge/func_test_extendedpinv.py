from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
def test_extendedpinv(self):
    X = standard_normal((40, 10))
    np_inv = np.linalg.pinv(X)
    np_sing_vals = np.linalg.svd(X, 0, 0)
    sm_inv, sing_vals = pinv_extended(X)
    assert_almost_equal(np_inv, sm_inv)
    assert_almost_equal(np_sing_vals, sing_vals)