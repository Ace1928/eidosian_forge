from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_coverage(self, sort):
    idx = DatetimeIndex(['2000-01-03', '2000-01-01', '2000-01-02'])
    ordered = DatetimeIndex(idx.sort_values(), freq='infer')
    result = ordered.union(idx, sort=sort)
    tm.assert_index_equal(result, ordered)
    result = ordered[:0].union(ordered, sort=sort)
    tm.assert_index_equal(result, ordered)
    assert result.freq == ordered.freq