from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_rank_multiindex():
    df = concat({'a': DataFrame({'col1': [3, 4], 'col2': [1, 2]}), 'b': DataFrame({'col3': [5, 6], 'col4': [7, 8]})}, axis=1)
    gb = df.groupby(level=0, axis=1)
    result = gb.rank(axis=1)
    expected = concat([df['a'].rank(axis=1), df['b'].rank(axis=1)], axis=1, keys=['a', 'b'])
    tm.assert_frame_equal(result, expected)