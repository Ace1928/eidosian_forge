import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('idx', [pd.Index([1, 1]), pd.Index(['a', 'a']), pd.Index([1.1, 1.1]), pd.PeriodIndex([pd.Period('2000', 'D')] * 2), pd.DatetimeIndex([pd.Timestamp('2000')] * 2), pd.TimedeltaIndex([pd.Timedelta('1D')] * 2), pd.CategoricalIndex(['a', 'a']), pd.IntervalIndex([pd.Interval(0, 1)] * 2), pd.MultiIndex.from_tuples([('a', 1), ('a', 1)])], ids=lambda x: type(x).__name__)
def test_raises_basic(idx):
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.Series(1, index=idx).set_flags(allows_duplicate_labels=False)
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame({'A': [1, 1]}, index=idx).set_flags(allows_duplicate_labels=False)
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame([[1, 2]], columns=idx).set_flags(allows_duplicate_labels=False)