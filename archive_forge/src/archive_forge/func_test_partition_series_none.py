from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, exp', [['partition', [('a', ' ', 'b c'), ('c', ' ', 'd e'), np.nan, ('f', ' ', 'g h'), None]], ['rpartition', [('a b', ' ', 'c'), ('c d', ' ', 'e'), np.nan, ('f g', ' ', 'h'), None]]])
def test_partition_series_none(any_string_dtype, method, exp):
    s = Series(['a b c', 'c d e', np.nan, 'f g h', None], dtype=any_string_dtype)
    result = getattr(s.str, method)(expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)