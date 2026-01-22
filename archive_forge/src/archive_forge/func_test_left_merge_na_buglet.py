import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_left_merge_na_buglet(self):
    left = DataFrame({'id': list('abcde'), 'v1': np.random.default_rng(2).standard_normal(5), 'v2': np.random.default_rng(2).standard_normal(5), 'dummy': list('abcde'), 'v3': np.random.default_rng(2).standard_normal(5)}, columns=['id', 'v1', 'v2', 'dummy', 'v3'])
    right = DataFrame({'id': ['a', 'b', np.nan, np.nan, np.nan], 'sv3': [1.234, 5.678, np.nan, np.nan, np.nan]})
    result = merge(left, right, on='id', how='left')
    rdf = right.drop(['id'], axis=1)
    expected = left.join(rdf)
    tm.assert_frame_equal(result, expected)