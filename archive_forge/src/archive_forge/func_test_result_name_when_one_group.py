from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('name', ['some_name', None])
def test_result_name_when_one_group(name):
    ser = Series([1, 2], name=name)
    result = ser.groupby(['a', 'a'], group_keys=False).apply(lambda x: x)
    expected = Series([1, 2], name=name)
    tm.assert_series_equal(result, expected)