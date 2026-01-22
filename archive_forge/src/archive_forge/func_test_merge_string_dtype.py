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
@pytest.mark.parametrize('expected_data, how', [(['a', 'b'], 'outer'), ([], 'inner'), (['b'], 'right'), (['a'], 'left')])
def test_merge_string_dtype(self, how, expected_data, any_string_dtype):
    d1 = DataFrame([('a',)], columns=['id'], dtype=any_string_dtype)
    d2 = DataFrame([('b',)], columns=['id'], dtype=any_string_dtype)
    result = merge(d1, d2, how=how)
    exp_idx = RangeIndex(len(expected_data))
    expected = DataFrame(expected_data, index=exp_idx, columns=['id'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)