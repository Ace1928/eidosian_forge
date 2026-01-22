import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concatlike_same_dtypes(self, item):
    typ1, vals1 = item
    vals2 = vals1
    vals3 = vals1
    if typ1 == 'category':
        exp_data = Categorical(list(vals1) + list(vals2))
        exp_data3 = Categorical(list(vals1) + list(vals2) + list(vals3))
    else:
        exp_data = vals1 + vals2
        exp_data3 = vals1 + vals2 + vals3
    res = Index(vals1).append(Index(vals2))
    exp = Index(exp_data)
    tm.assert_index_equal(res, exp)
    res = Index(vals1).append([Index(vals2), Index(vals3)])
    exp = Index(exp_data3)
    tm.assert_index_equal(res, exp)
    i1 = Index(vals1, name='x')
    i2 = Index(vals2, name='y')
    res = i1.append(i2)
    exp = Index(exp_data)
    tm.assert_index_equal(res, exp)
    i1 = Index(vals1, name='x')
    i2 = Index(vals2, name='x')
    res = i1.append(i2)
    exp = Index(exp_data, name='x')
    tm.assert_index_equal(res, exp)
    with pytest.raises(TypeError, match='all inputs must be Index'):
        Index(vals1).append(vals2)
    with pytest.raises(TypeError, match='all inputs must be Index'):
        Index(vals1).append([Index(vals2), vals3])
    res = Series(vals1)._append(Series(vals2), ignore_index=True)
    exp = Series(exp_data)
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = Series(vals1)._append([Series(vals2), Series(vals3)], ignore_index=True)
    exp = Series(exp_data3)
    tm.assert_series_equal(res, exp)
    res = pd.concat([Series(vals1), Series(vals2), Series(vals3)], ignore_index=True)
    tm.assert_series_equal(res, exp)
    s1 = Series(vals1, name='x')
    s2 = Series(vals2, name='y')
    res = s1._append(s2, ignore_index=True)
    exp = Series(exp_data)
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = pd.concat([s1, s2], ignore_index=True)
    tm.assert_series_equal(res, exp, check_index_type=True)
    s1 = Series(vals1, name='x')
    s2 = Series(vals2, name='x')
    res = s1._append(s2, ignore_index=True)
    exp = Series(exp_data, name='x')
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = pd.concat([s1, s2], ignore_index=True)
    tm.assert_series_equal(res, exp, check_index_type=True)
    msg = "cannot concatenate object of type '.+'; only Series and DataFrame objs are valid"
    with pytest.raises(TypeError, match=msg):
        Series(vals1)._append(vals2)
    with pytest.raises(TypeError, match=msg):
        Series(vals1)._append([Series(vals2), vals3])
    with pytest.raises(TypeError, match=msg):
        pd.concat([Series(vals1), vals2])
    with pytest.raises(TypeError, match=msg):
        pd.concat([Series(vals1), Series(vals2), vals3])