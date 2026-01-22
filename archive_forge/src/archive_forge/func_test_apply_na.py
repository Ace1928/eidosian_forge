from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna', [True, False])
def test_apply_na(dropna):
    df = DataFrame({'grp': [1, 1, 2, 2], 'y': [1, 0, 2, 5], 'z': [1, 2, np.nan, np.nan]})
    dfgrp = df.groupby('grp', dropna=dropna)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = dfgrp.apply(lambda grp_df: grp_df.nlargest(1, 'z'))
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = dfgrp.apply(lambda x: x.sort_values('z', ascending=False).head(1))
    tm.assert_frame_equal(result, expected)