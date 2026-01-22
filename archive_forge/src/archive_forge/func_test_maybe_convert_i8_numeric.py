from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('make_key', [lambda breaks: breaks, list], ids=['lambda', 'list'])
def test_maybe_convert_i8_numeric(self, make_key, any_real_numpy_dtype):
    breaks = np.arange(5, dtype=any_real_numpy_dtype)
    index = IntervalIndex.from_breaks(breaks)
    key = make_key(breaks)
    result = index._maybe_convert_i8(key)
    kind = breaks.dtype.kind
    expected_dtype = {'i': np.int64, 'u': np.uint64, 'f': np.float64}[kind]
    expected = Index(key, dtype=expected_dtype)
    tm.assert_index_equal(result, expected)