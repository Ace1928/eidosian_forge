import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
@pytest.mark.smoke
def test_coint_johansen_0lag(reset_randomstate):
    x_diff = np.random.normal(0, 1, 1000)
    x = pd.Series(np.cumsum(x_diff))
    e1 = np.random.normal(0, 1, 1000)
    y = x + 5 + e1
    data = pd.concat([x, y], axis=1)
    result = coint_johansen(data, det_order=-1, k_ar_diff=0)
    assert result.eig.shape == (2,)