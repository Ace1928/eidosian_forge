from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype, val', [(object, '01'), ('int64', 1)])
def test_specify_dtype_for_index_col(all_parsers, dtype, val, request):
    data = 'a,b\n01,2'
    parser = all_parsers
    if dtype == object and parser.engine == 'pyarrow':
        request.applymarker(pytest.mark.xfail(reason='Cannot disable type-inference for pyarrow engine'))
    result = parser.read_csv(StringIO(data), index_col='a', dtype={'a': dtype})
    expected = DataFrame({'b': [2]}, index=Index([val], name='a'))
    tm.assert_frame_equal(result, expected)