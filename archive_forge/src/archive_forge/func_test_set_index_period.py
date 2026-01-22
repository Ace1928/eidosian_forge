from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_period(self):
    df = DataFrame(np.random.default_rng(2).random(6))
    idx1 = period_range('2011-01-01', periods=3, freq='M')
    idx1 = idx1.append(idx1)
    idx2 = period_range('2013-01-01 09:00', periods=2, freq='h')
    idx2 = idx2.append(idx2).append(idx2)
    idx3 = period_range('2005', periods=6, freq='Y')
    df = df.set_index(idx1)
    df = df.set_index(idx2, append=True)
    df = df.set_index(idx3, append=True)
    expected1 = period_range('2011-01-01', periods=3, freq='M')
    expected2 = period_range('2013-01-01 09:00', periods=2, freq='h')
    tm.assert_index_equal(df.index.levels[0], expected1)
    tm.assert_index_equal(df.index.levels[1], expected2)
    tm.assert_index_equal(df.index.levels[2], idx3)
    tm.assert_index_equal(df.index.get_level_values(0), idx1)
    tm.assert_index_equal(df.index.get_level_values(1), idx2)
    tm.assert_index_equal(df.index.get_level_values(2), idx3)