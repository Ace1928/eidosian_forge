import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data1,data2', [(range(3), range(1, 4)), (list('abc'), list('xyz')), (list('áàä'), list('éèë')), (list('áàä'), list(b'aaa')), (range(3), range(4))])
def test_series_not_equal_value_mismatch(data1, data2):
    _assert_not_series_equal_both(Series(data1), Series(data2))