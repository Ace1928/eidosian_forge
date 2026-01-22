from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_timezone():
    result = DataFrame({'foo': [datetime(2000, 1, 1)]})
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        result.at[0, 'foo'] = datetime(2000, 1, 2, tzinfo=timezone.utc)
    expected = DataFrame({'foo': [datetime(2000, 1, 2, tzinfo=timezone.utc)]}, dtype=object)
    tm.assert_frame_equal(result, expected)