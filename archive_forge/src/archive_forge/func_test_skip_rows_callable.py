from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('kwargs,expected', [({}, DataFrame({'1': [3, 5]})), ({'header': 0, 'names': ['foo']}, DataFrame({'foo': [3, 5]}))])
def test_skip_rows_callable(all_parsers, kwargs, expected):
    parser = all_parsers
    data = 'a\n1\n2\n3\n4\n5'
    result = parser.read_csv(StringIO(data), skiprows=lambda x: x % 2 == 0, **kwargs)
    tm.assert_frame_equal(result, expected)