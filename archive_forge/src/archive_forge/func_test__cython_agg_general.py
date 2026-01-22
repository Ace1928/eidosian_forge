import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('op, targop', [('mean', np.mean), ('median', np.median), ('var', np.var), ('sum', np.sum), ('prod', np.prod), ('min', np.min), ('max', np.max), ('first', lambda x: x.iloc[0]), ('last', lambda x: x.iloc[-1])])
def test__cython_agg_general(op, targop):
    df = DataFrame(np.random.default_rng(2).standard_normal(1000))
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    result = df.groupby(labels)._cython_agg_general(op, alt=None, numeric_only=True)
    warn = FutureWarning if targop in com._cython_table else None
    msg = f'using DataFrameGroupBy.{op}'
    with tm.assert_produces_warning(warn, match=msg):
        expected = df.groupby(labels).agg(targop)
    tm.assert_frame_equal(result, expected)