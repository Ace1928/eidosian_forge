from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'])
@pytest.mark.parametrize('upper', [True, False])
def test_rank_avg_even_vals(dtype, upper):
    if upper:
        dtype = dtype[0].upper() + dtype[1:]
        dtype = dtype.replace('Ui', 'UI')
    df = DataFrame({'key': ['a'] * 4, 'val': [1] * 4})
    df['val'] = df['val'].astype(dtype)
    assert df['val'].dtype == dtype
    result = df.groupby('key').rank()
    exp_df = DataFrame([2.5, 2.5, 2.5, 2.5], columns=['val'])
    if upper:
        exp_df = exp_df.astype('Float64')
    tm.assert_frame_equal(result, exp_df)