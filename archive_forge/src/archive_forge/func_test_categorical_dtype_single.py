from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [{'b': 'category'}, {1: 'category'}])
def test_categorical_dtype_single(all_parsers, dtype, request):
    parser = all_parsers
    data = 'a,b,c\n1,a,3.4\n1,a,3.4\n2,b,4.5'
    expected = DataFrame({'a': [1, 1, 2], 'b': Categorical(['a', 'a', 'b']), 'c': [3.4, 3.4, 4.5]})
    if parser.engine == 'pyarrow':
        mark = pytest.mark.xfail(strict=False, reason='Flaky test sometimes gives object dtype instead of Categorical')
        request.applymarker(mark)
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)