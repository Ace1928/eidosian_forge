from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_ignore_index(self, sort):
    frame1 = DataFrame({'test1': ['a', 'b', 'c'], 'test2': [1, 2, 3], 'test3': [4.5, 3.2, 1.2]})
    frame2 = DataFrame({'test3': [5.2, 2.2, 4.3]})
    frame1.index = Index(['x', 'y', 'z'])
    frame2.index = Index(['x', 'y', 'q'])
    v1 = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)
    nan = np.nan
    expected = DataFrame([[nan, nan, nan, 4.3], ['a', 1, 4.5, 5.2], ['b', 2, 3.2, 2.2], ['c', 3, 1.2, nan]], index=Index(['q', 'x', 'y', 'z']))
    if not sort:
        expected = expected.loc[['x', 'y', 'z', 'q']]
    tm.assert_frame_equal(v1, expected)