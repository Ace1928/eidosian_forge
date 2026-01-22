from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
def test_where_alignment(self, where_frame, float_string_frame):

    def _check_align(df, cond, other, check_dtypes=True):
        rs = df.where(cond, other)
        for i, k in enumerate(rs.columns):
            result = rs[k]
            d = df[k].values
            c = cond[k].reindex(df[k].index).fillna(False).values
            if is_scalar(other):
                o = other
            elif isinstance(other, np.ndarray):
                o = Series(other[:, i], index=result.index).values
            else:
                o = other[k].values
            new_values = d if c.all() else np.where(c, d, o)
            expected = Series(new_values, index=result.index, name=k)
            tm.assert_series_equal(result, expected, check_dtype=False)
        if check_dtypes and (not isinstance(other, np.ndarray)):
            assert (rs.dtypes == df.dtypes).all()
    df = where_frame
    if df is float_string_frame:
        msg = "'>' not supported between instances of 'str' and 'int'"
        with pytest.raises(TypeError, match=msg):
            df > 0
        return
    cond = (df > 0)[1:]
    _check_align(df, cond, _safe_add(df))
    cond = df > 0
    _check_align(df, cond, _safe_add(df).values)
    cond = df > 0
    check_dtypes = all((not issubclass(s.type, np.integer) for s in df.dtypes))
    _check_align(df, cond, np.nan, check_dtypes=check_dtypes)