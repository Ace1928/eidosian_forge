import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_validate_sum_initial():
    ser = Series([1, 2])
    msg = "the 'initial' parameter is not supported in the pandas implementation of sum\\(\\)"
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, initial=10)