import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_merge_na_keys(self):
    data = [[1950, 'A', 1.5], [1950, 'B', 1.5], [1955, 'B', 1.5], [1960, 'B', np.nan], [1970, 'B', 4.0], [1950, 'C', 4.0], [1960, 'C', np.nan], [1965, 'C', 3.0], [1970, 'C', 4.0]]
    frame = DataFrame(data, columns=['year', 'panel', 'data'])
    other_data = [[1960, 'A', np.nan], [1970, 'A', np.nan], [1955, 'A', np.nan], [1965, 'A', np.nan], [1965, 'B', np.nan], [1955, 'C', np.nan]]
    other = DataFrame(other_data, columns=['year', 'panel', 'data'])
    result = frame.merge(other, how='outer')
    expected = frame.fillna(-999).merge(other.fillna(-999), how='outer')
    expected = expected.replace(-999, np.nan)
    tm.assert_frame_equal(result, expected)