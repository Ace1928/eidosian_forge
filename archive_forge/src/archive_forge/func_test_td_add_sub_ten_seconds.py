from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('ten_seconds', [Timedelta(10, unit='s'), timedelta(seconds=10), np.timedelta64(10, 's'), np.timedelta64(10000000000, 'ns'), offsets.Second(10)])
def test_td_add_sub_ten_seconds(self, ten_seconds):
    base = Timestamp('20130101 09:01:12.123456')
    expected_add = Timestamp('20130101 09:01:22.123456')
    expected_sub = Timestamp('20130101 09:01:02.123456')
    result = base + ten_seconds
    assert result == expected_add
    result = base - ten_seconds
    assert result == expected_sub