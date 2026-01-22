import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_stable_categorial(self):
    df = DataFrame({'x': Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
    expected = df.copy()
    sorted_df = df.sort_values('x', kind='mergesort')
    tm.assert_frame_equal(sorted_df, expected)