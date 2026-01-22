import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('array,expected', [([0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j], Series([False, True, True, False, True, True, True], dtype=bool))])
def test_isin_complex_numbers(array, expected):
    result = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    tm.assert_series_equal(result, expected)