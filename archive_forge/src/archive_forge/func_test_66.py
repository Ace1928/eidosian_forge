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
def test_66(self):
    nan = np.nan
    test_res = tools.nan_dot(self.mx_6, self.mx_6)
    expected_res = np.array([[7.0, 10.0], [15.0, 22.0]])
    assert_array_equal(test_res, expected_res)