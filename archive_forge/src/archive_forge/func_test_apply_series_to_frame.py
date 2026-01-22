from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_series_to_frame():

    def f(piece):
        with np.errstate(invalid='ignore'):
            logged = np.log(piece)
        return DataFrame({'value': piece, 'demeaned': piece - piece.mean(), 'logged': logged})
    dr = bdate_range('1/1/2000', periods=100)
    ts = Series(np.random.default_rng(2).standard_normal(100), index=dr)
    grouped = ts.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(f)
    assert isinstance(result, DataFrame)
    assert not hasattr(result, 'name')
    tm.assert_index_equal(result.index, ts.index)