import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.fixture(params=[DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 0]), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 1]), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=['C', 'C']), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1.0, 0]), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0.0, 1]), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=['C', 1]), DataFrame([[2.0, 4.0], [1.0, 2.0], [5.0, 2.0], [8.0, 1.0]], columns=[1, 0.0]), DataFrame([[2, 4.0], [1, 2.0], [5, 2.0], [8, 1.0]], columns=[0, 1.0]), DataFrame([[2, 4], [1, 2], [5, 2], [8, 1.0]], columns=[1.0, 'X'])])
def pairwise_frames(request):
    """Pairwise frames test_pairwise"""
    return request.param