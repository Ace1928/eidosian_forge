import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('op, targop', [('mean', np.mean), ('median', lambda x: np.median(x) if len(x) > 0 else np.nan), ('var', lambda x: np.var(x, ddof=1)), ('min', np.min), ('max', np.max)])
def test_cython_agg_empty_buckets(op, targop, observed):
    df = DataFrame([11, 12, 13])
    grps = range(0, 55, 5)
    g = df.groupby(pd.cut(df[0], grps), observed=observed)
    result = g._cython_agg_general(op, alt=None, numeric_only=True)
    g = df.groupby(pd.cut(df[0], grps), observed=observed)
    expected = g.agg(lambda x: targop(x))
    tm.assert_frame_equal(result, expected)