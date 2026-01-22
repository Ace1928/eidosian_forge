import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Hour
def test_zero_length_input_index(self, sort):
    index_1 = timedelta_range('1 day', periods=0, freq='h')
    index_2 = timedelta_range('1 day', periods=3, freq='h')
    result = index_1.intersection(index_2, sort=sort)
    assert index_1 is not result
    assert index_2 is not result
    tm.assert_copy(result, index_1)