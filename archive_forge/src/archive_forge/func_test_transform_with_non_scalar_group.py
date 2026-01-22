import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_with_non_scalar_group():
    cols = MultiIndex.from_tuples([('syn', 'A'), ('foo', 'A'), ('non', 'A'), ('syn', 'C'), ('foo', 'C'), ('non', 'C'), ('syn', 'T'), ('foo', 'T'), ('non', 'T'), ('syn', 'G'), ('foo', 'G'), ('non', 'G')])
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (4, 12)), columns=cols, index=['A', 'C', 'G', 'T'])
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(axis=1, level=1)
    msg = 'transform must return a scalar value for each group.*'
    with pytest.raises(ValueError, match=msg):
        gb.transform(lambda z: z.div(z.sum(axis=1), axis=0))