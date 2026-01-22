import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multi_axis_dups(self):
    df = DataFrame(np.arange(25.0).reshape(5, 5), index=['a', 'b', 'c', 'd', 'e'], columns=['A', 'B', 'C', 'D', 'E'])
    z = df[['A', 'C', 'A']].copy()
    expected = z.loc[['a', 'c', 'a']]
    df = DataFrame(np.arange(25.0).reshape(5, 5), index=['a', 'b', 'c', 'd', 'e'], columns=['A', 'B', 'C', 'D', 'E'])
    z = df[['A', 'C', 'A']]
    result = z.loc[['a', 'c', 'a']]
    tm.assert_frame_equal(result, expected)