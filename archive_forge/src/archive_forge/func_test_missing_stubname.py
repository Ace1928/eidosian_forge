import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['O', 'string'])
def test_missing_stubname(self, dtype):
    df = DataFrame({'id': ['1', '2'], 'a-1': [100, 200], 'a-2': [300, 400]})
    df = df.astype({'id': dtype})
    result = wide_to_long(df, stubnames=['a', 'b'], i='id', j='num', sep='-')
    index = Index([('1', 1), ('2', 1), ('1', 2), ('2', 2)], name=('id', 'num'))
    expected = DataFrame({'a': [100, 200, 300, 400], 'b': [np.nan] * 4}, index=index)
    new_level = expected.index.levels[0].astype(dtype)
    expected.index = expected.index.set_levels(new_level, level=0)
    tm.assert_frame_equal(result, expected)