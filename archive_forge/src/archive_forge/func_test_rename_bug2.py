from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_bug2(self):
    df = DataFrame(data=np.arange(3), index=[(0, 0), (1, 1), (2, 2)], columns=['a'])
    df = df.rename({(1, 1): (5, 4)}, axis='index')
    expected = DataFrame(data=np.arange(3), index=[(0, 0), (5, 4), (2, 2)], columns=['a'])
    tm.assert_frame_equal(df, expected)