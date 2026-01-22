from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_index_singlekey_right_vs_left(self):
    left = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'], 'v1': np.random.default_rng(2).standard_normal(7)})
    right = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)}, index=['d', 'b', 'c', 'a'])
    merged1 = merge(left, right, left_on='key', right_index=True, how='left', sort=False)
    merged2 = merge(right, left, right_on='key', left_index=True, how='right', sort=False)
    tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])
    merged1 = merge(left, right, left_on='key', right_index=True, how='left', sort=True)
    merged2 = merge(right, left, right_on='key', left_index=True, how='right', sort=True)
    tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])