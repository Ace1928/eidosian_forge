import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_contains_float64_nans(self):
    index = Index([1.0, 2.0, np.nan], dtype=np.float64)
    assert np.nan in index