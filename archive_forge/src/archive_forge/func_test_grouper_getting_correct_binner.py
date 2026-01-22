from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouper_getting_correct_binner(self):
    df = DataFrame({'A': 1}, index=MultiIndex.from_product([list('ab'), date_range('20130101', periods=80)], names=['one', 'two']))
    result = df.groupby([Grouper(level='one'), Grouper(level='two', freq='ME')]).sum()
    expected = DataFrame({'A': [31, 28, 21, 31, 28, 21]}, index=MultiIndex.from_product([list('ab'), date_range('20130101', freq='ME', periods=3)], names=['one', 'two']))
    tm.assert_frame_equal(result, expected)