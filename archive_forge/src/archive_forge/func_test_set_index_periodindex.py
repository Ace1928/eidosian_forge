from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_periodindex(self):
    df = DataFrame(np.random.default_rng(2).random(6))
    idx1 = period_range('2011/01/01', periods=6, freq='M')
    idx2 = period_range('2013', periods=6, freq='Y')
    df = df.set_index(idx1)
    tm.assert_index_equal(df.index, idx1)
    df = df.set_index(idx2)
    tm.assert_index_equal(df.index, idx2)