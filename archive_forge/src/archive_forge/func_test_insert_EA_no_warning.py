import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_EA_no_warning(self):
    df = DataFrame(np.random.default_rng(2).integers(0, 100, size=(3, 100)), dtype='Int64')
    with tm.assert_produces_warning(None):
        df['a'] = np.array([1, 2, 3])