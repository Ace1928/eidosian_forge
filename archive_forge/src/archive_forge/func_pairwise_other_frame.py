import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.fixture
def pairwise_other_frame():
    """Pairwise other frame for test_pairwise"""
    return DataFrame([[None, 1, 1], [None, 1, 2], [None, 3, 2], [None, 8, 1]], columns=['Y', 'Z', 'X'])