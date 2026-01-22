import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_frame_getitem_multicolumn_empty_level():
    df = DataFrame({'a': ['1', '2', '3'], 'b': ['2', '3', '4']})
    df.columns = [['level1 item1', 'level1 item2'], ['', 'level2 item2'], ['level3 item1', 'level3 item2']]
    result = df['level1 item1']
    expected = DataFrame([['1'], ['2'], ['3']], index=df.index, columns=['level3 item1'])
    tm.assert_frame_equal(result, expected)