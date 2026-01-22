import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_rename_multiindex_with_duplicates(self):
    mi = MultiIndex.from_tuples([('A', 'cat'), ('B', 'cat'), ('B', 'cat')])
    df = DataFrame(index=mi)
    df = df.rename(index={'A': 'Apple'}, level=0)
    mi2 = MultiIndex.from_tuples([('Apple', 'cat'), ('B', 'cat'), ('B', 'cat')])
    expected = DataFrame(index=mi2)
    tm.assert_frame_equal(df, expected)