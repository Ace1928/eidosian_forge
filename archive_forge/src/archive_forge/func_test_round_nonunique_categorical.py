import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_round_nonunique_categorical(self):
    idx = pd.CategoricalIndex(['low'] * 3 + ['hi'] * 3)
    df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=list('abc'))
    expected = df.round(3)
    expected.index = idx
    df_categorical = df.copy().set_index(idx)
    assert df_categorical.shape == (6, 3)
    result = df_categorical.round(3)
    assert result.shape == (6, 3)
    tm.assert_frame_equal(result, expected)