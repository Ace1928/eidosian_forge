import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set int into string")
def test_rhs_alignment(self):

    def run_tests(df, rhs, right_loc, right_iloc):
        lbl_one, idx_one, slice_one = (list('bcd'), [1, 2, 3], slice(1, 4))
        lbl_two, idx_two, slice_two = (['joe', 'jolie'], [1, 2], slice(1, 3))
        left = df.copy()
        left.loc[lbl_one, lbl_two] = rhs
        tm.assert_frame_equal(left, right_loc)
        left = df.copy()
        left.iloc[idx_one, idx_two] = rhs
        tm.assert_frame_equal(left, right_iloc)
        left = df.copy()
        left.iloc[slice_one, slice_two] = rhs
        tm.assert_frame_equal(left, right_iloc)
    xs = np.arange(20).reshape(5, 4)
    cols = ['jim', 'joe', 'jolie', 'joline']
    df = DataFrame(xs, columns=cols, index=list('abcde'), dtype='int64')
    rhs = -2 * df.iloc[3:0:-1, 2:0:-1]
    right_iloc = df.copy()
    right_iloc['joe'] = [1, 14, 10, 6, 17]
    right_iloc['jolie'] = [2, 13, 9, 5, 18]
    right_iloc.iloc[1:4, 1:3] *= -2
    right_loc = df.copy()
    right_loc.iloc[1:4, 1:3] *= -2
    run_tests(df, rhs, right_loc, right_iloc)
    for frame in [df, rhs, right_loc, right_iloc]:
        frame['joe'] = frame['joe'].astype('float64')
        frame['jolie'] = frame['jolie'].map(lambda x: f'@{x}')
    right_iloc['joe'] = [1.0, '@-28', '@-20', '@-12', 17.0]
    right_iloc['jolie'] = ['@2', -26.0, -18.0, -10.0, '@18']
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        run_tests(df, rhs, right_loc, right_iloc)