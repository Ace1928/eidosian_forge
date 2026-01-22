from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('sort', [False, None])
def test_intersection_duplicates(self, sort):
    idx1 = Index([Timestamp('2019-12-13'), Timestamp('2019-12-12'), Timestamp('2019-12-12')])
    result = idx1.intersection(idx1, sort=sort)
    expected = Index([Timestamp('2019-12-13'), Timestamp('2019-12-12')])
    tm.assert_index_equal(result, expected)