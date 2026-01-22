import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
def test_searchsorted_nan(self, dtype):
    arr = pd.array(range(10), dtype=dtype)
    assert arr.searchsorted(np.nan, side='left') == 10
    assert arr.searchsorted(np.nan, side='right') == 10