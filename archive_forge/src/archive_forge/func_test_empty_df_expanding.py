import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('expander', [1, pytest.param('ls', marks=pytest.mark.xfail(reason='GH#16425 expanding with offset not supported'))])
def test_empty_df_expanding(expander):
    expected = DataFrame()
    result = DataFrame().expanding(expander).sum()
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()
    tm.assert_frame_equal(result, expected)