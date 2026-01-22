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
def test_recipr(self):
    X = np.array([[2, 1], [-1, 0]])
    Y = tools.recipr(X)
    assert_almost_equal(Y, np.array([[0.5, 1], [0, 0]]))