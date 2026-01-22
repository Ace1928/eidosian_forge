from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_nan_index2(self):
    df = DataFrame({'A': list('aaaabbbb'), 'B': range(8), 'C': range(8)})
    df = df.astype({'B': 'float'})
    df.iloc[3, 1] = np.nan
    left = df.set_index(['A', 'B']).unstack(0)
    vals = [[3, 0, 1, 2, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7]]
    vals = list(map(list, zip(*vals)))
    idx = Index([np.nan, 0, 1, 2, 4, 5, 6, 7], name='B')
    cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
    right = DataFrame(vals, columns=cols, index=idx)
    tm.assert_frame_equal(left, right)
    df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
    df = df.astype({'B': 'float'})
    df.iloc[2, 1] = np.nan
    left = df.set_index(['A', 'B']).unstack(0)
    vals = [[2, np.nan], [0, 4], [1, 5], [np.nan, 6], [3, 7]]
    cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
    idx = Index([np.nan, 0, 1, 2, 3], name='B')
    right = DataFrame(vals, columns=cols, index=idx)
    tm.assert_frame_equal(left, right)
    df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
    df = df.astype({'B': 'float'})
    df.iloc[3, 1] = np.nan
    left = df.set_index(['A', 'B']).unstack(0)
    vals = [[3, np.nan], [0, 4], [1, 5], [2, 6], [np.nan, 7]]
    cols = MultiIndex(levels=[['C'], ['a', 'b']], codes=[[0, 0], [0, 1]], names=[None, 'A'])
    idx = Index([np.nan, 0, 1, 2, 3], name='B')
    right = DataFrame(vals, columns=cols, index=idx)
    tm.assert_frame_equal(left, right)