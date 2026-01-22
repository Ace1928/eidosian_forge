from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
def test_rank_does_not_mutate(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), dtype='float64')
    expected = df.copy()
    df.rank()
    result = df
    tm.assert_frame_equal(result, expected)