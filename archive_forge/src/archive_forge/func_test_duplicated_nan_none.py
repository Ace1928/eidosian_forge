import re
import sys
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason='GH#21720; nan/None falsely considered equal')
@pytest.mark.parametrize('keep, expected', [('first', Series([False, False, True, False, True])), ('last', Series([True, True, False, False, False])), (False, Series([True, True, True, False, True]))])
def test_duplicated_nan_none(keep, expected):
    df = DataFrame({'C': [np.nan, 3, 3, None, np.nan], 'x': 1}, dtype=object)
    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)