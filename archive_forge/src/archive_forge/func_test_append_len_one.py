import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ri', [RangeIndex(0, -1, -1), RangeIndex(0, 1, 1), RangeIndex(1, 3, 2), RangeIndex(0, -1, -2), RangeIndex(-3, -5, -2)])
def test_append_len_one(self, ri):
    result = ri.append([])
    tm.assert_index_equal(result, ri, exact=True)