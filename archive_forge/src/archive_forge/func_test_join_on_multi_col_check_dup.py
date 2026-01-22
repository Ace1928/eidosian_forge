from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_on_multi_col_check_dup():
    left = DataFrame({'a': ['a', 'a', 'b', 'b'], 'b': [0, 1, 0, 1], 'c': ['cat', 'dog', 'weasel', 'horse']}, index=range(4)).set_index(['a', 'b'])
    right = DataFrame({'a': ['a', 'a', 'b'], 'b': [0, 1, 0], 'd': ['meow', 'bark', 'um... weasel noise?']}, index=range(3)).set_index(['a', 'b'])
    expected_multi = DataFrame({'a': ['a', 'a', 'b'], 'b': [0, 1, 0], 'c': ['cat', 'dog', 'weasel'], 'd': ['meow', 'bark', 'um... weasel noise?']}, index=range(3)).set_index(['a', 'b'])
    result = left.join(right, how='inner', validate='1:1')
    tm.assert_frame_equal(result, expected_multi)