import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'America/New_York'])
def test_transpose_preserves_dtindex_equality_with_dst(self, tz):
    idx = date_range('20161101', '20161130', freq='4h', tz=tz)
    df = DataFrame({'a': range(len(idx)), 'b': range(len(idx))}, index=idx)
    result = df.T == df.T
    expected = DataFrame(True, index=list('ab'), columns=idx)
    tm.assert_frame_equal(result, expected)