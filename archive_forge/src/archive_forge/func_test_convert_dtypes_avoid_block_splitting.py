import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_convert_dtypes_avoid_block_splitting(self):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': 'a'})
    result = df.convert_dtypes(convert_integer=False)
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': pd.Series(['a'] * 3, dtype='string[python]')})
    tm.assert_frame_equal(result, expected)
    assert result._mgr.nblocks == 2