from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_cross_type():
    vals = np.hstack((np.random.default_rng(2).integers(0, 5, (100, 2)), np.random.default_rng(2).integers(0, 2, (100, 2)))).astype('float64')
    df = DataFrame(vals, columns=['a', 'b', 'c', 'd'])
    df[df == 2] = np.nan
    expected = df.groupby(['c', 'd']).count()
    for t in ['float32', 'object']:
        df['a'] = df['a'].astype(t)
        df['b'] = df['b'].astype(t)
        result = df.groupby(['c', 'd']).count()
        tm.assert_frame_equal(result, expected)