from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ties_method', ['average', 'min', 'max', 'first', 'dense'])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('na_option', ['keep', 'top', 'bottom'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('vals', [['bar', 'bar', 'foo', 'bar', 'baz'], ['bar', np.nan, 'foo', np.nan, 'baz']])
def test_rank_object_dtype(ties_method, ascending, na_option, pct, vals):
    df = DataFrame({'key': ['foo'] * 5, 'val': vals})
    mask = df['val'].isna()
    gb = df.groupby('key')
    res = gb.rank(method=ties_method, ascending=ascending, na_option=na_option, pct=pct)
    if mask.any():
        df2 = DataFrame({'key': ['foo'] * 5, 'val': [0, np.nan, 2, np.nan, 1]})
    else:
        df2 = DataFrame({'key': ['foo'] * 5, 'val': [0, 0, 2, 0, 1]})
    gb2 = df2.groupby('key')
    alt = gb2.rank(method=ties_method, ascending=ascending, na_option=na_option, pct=pct)
    tm.assert_frame_equal(res, alt)