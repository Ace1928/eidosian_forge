from datetime import (
from pandas import (
import pandas._testing as tm
def test_is_unique_monotonic(rand_series_with_duplicate_datetimeindex):
    index = rand_series_with_duplicate_datetimeindex.index
    assert not index.is_unique