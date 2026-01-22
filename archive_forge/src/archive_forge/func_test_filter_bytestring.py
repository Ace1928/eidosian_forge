import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('name', ['a', 'a'])
def test_filter_bytestring(self, name):
    df = DataFrame({b'a': [1, 2], b'b': [3, 4]})
    expected = DataFrame({b'a': [1, 2]})
    tm.assert_frame_equal(df.filter(like=name), expected)
    tm.assert_frame_equal(df.filter(regex=name), expected)