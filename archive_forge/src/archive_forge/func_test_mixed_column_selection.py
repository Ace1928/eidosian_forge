import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mixed_column_selection(self):
    dfbool = DataFrame({'one': Series([True, True, False], index=['a', 'b', 'c']), 'two': Series([False, False, True, False], index=['a', 'b', 'c', 'd']), 'three': Series([False, True, True, True], index=['a', 'b', 'c', 'd'])})
    expected = pd.concat([dfbool['one'], dfbool['three'], dfbool['one']], axis=1)
    result = dfbool[['one', 'three', 'one']]
    tm.assert_frame_equal(result, expected)