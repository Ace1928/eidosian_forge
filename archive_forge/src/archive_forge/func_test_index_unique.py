from datetime import (
from pandas import (
import pandas._testing as tm
def test_index_unique(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    index = dups.index
    uniques = index.unique()
    expected = DatetimeIndex([datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5)])
    assert uniques.dtype == 'M8[ns]'
    tm.assert_index_equal(uniques, expected)
    assert index.nunique() == 4
    assert isinstance(uniques, DatetimeIndex)
    dups_local = index.tz_localize('US/Eastern')
    dups_local.name = 'foo'
    result = dups_local.unique()
    expected = DatetimeIndex(expected, name='foo')
    expected = expected.tz_localize('US/Eastern')
    assert result.tz is not None
    assert result.name == 'foo'
    tm.assert_index_equal(result, expected)