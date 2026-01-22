import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_irregular_datetime_index_with_nan(self):
    index = pd.to_datetime(['2012-01-01', '2012-01-02', '2012-01-03', None])
    df = DataFrame(range(len(index)), index=index)
    expected = DataFrame(range(len(index[:3])), index=index[:3])
    with pytest.raises(KeyError, match='non-existing keys is not allowed'):
        df['2012-01-01':'2012-01-04']
    result = df['2012-01-01':'2012-01-03 00:00:00.000000000']
    tm.assert_frame_equal(result, expected)