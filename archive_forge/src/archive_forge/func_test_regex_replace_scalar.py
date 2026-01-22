from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [{'a': list('ab..'), 'b': list('efgh')}, {'a': list('ab..'), 'b': list(range(4))}])
@pytest.mark.parametrize('to_replace,value', [('\\s*\\.\\s*', np.nan), ('\\s*(\\.)\\s*', '\\1\\1\\1')])
@pytest.mark.parametrize('compile_regex', [True, False])
@pytest.mark.parametrize('regex_kwarg', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
def test_regex_replace_scalar(self, data, to_replace, value, compile_regex, regex_kwarg, inplace):
    df = DataFrame(data)
    expected = df.copy()
    if compile_regex:
        to_replace = re.compile(to_replace)
    if regex_kwarg:
        regex = to_replace
        to_replace = None
    else:
        regex = True
    result = df.replace(to_replace, value, inplace=inplace, regex=regex)
    if inplace:
        assert result is None
        result = df
    if value is np.nan:
        expected_replace_val = np.nan
    else:
        expected_replace_val = '...'
    expected.loc[expected['a'] == '.', 'a'] = expected_replace_val
    tm.assert_frame_equal(result, expected)