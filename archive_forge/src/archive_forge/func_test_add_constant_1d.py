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
def test_add_constant_1d(self):
    x = np.arange(1, 5)
    x = tools.add_constant(x)
    y = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
    assert_equal(x, y)