from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('na_filter', [True, False])
def test_na_values_with_dtype_str_and_na_filter(all_parsers, na_filter):
    parser = all_parsers
    data = 'a,b,c\n1,,3\n4,5,6'
    empty = np.nan if na_filter else ''
    expected = DataFrame({'a': ['1', '4'], 'b': [empty, '5'], 'c': ['3', '6']})
    result = parser.read_csv(StringIO(data), na_filter=na_filter, dtype=str)
    tm.assert_frame_equal(result, expected)