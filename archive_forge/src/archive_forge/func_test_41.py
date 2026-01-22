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
def test_41(self):
    nan = np.nan
    test_res = tools.nan_dot(self.mx_4, self.mx_1)
    expected_res = np.array([[nan, 1.0], [nan, 1.0]])
    assert_array_equal(test_res, expected_res)