import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_with_nan_in_index(nulls_fixture):
    mi = MultiIndex.from_tuples([('blah', nulls_fixture)], names=['name', 'date'])
    msg = "labels \\[Timestamp\\('2001-01-01 00:00:00'\\)\\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop(pd.Timestamp('2001'), level='date')