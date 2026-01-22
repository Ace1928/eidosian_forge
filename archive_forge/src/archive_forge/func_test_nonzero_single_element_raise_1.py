from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [np.nan, pd.NaT, True, False])
def test_nonzero_single_element_raise_1(self, data):
    series = Series([data])
    msg = 'The truth value of a Series is ambiguous'
    with pytest.raises(ValueError, match=msg):
        bool(series)