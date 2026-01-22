import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_single_level_drop_partially_missing_elements():
    mi = MultiIndex.from_tuples([(1, 2), (2, 2), (3, 2)])
    msg = 'labels \\[4\\] not found in level'
    with pytest.raises(KeyError, match=msg):
        mi.drop(4, level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([1, 4], level=0)
    msg = 'labels \\[nan\\] not found in level'
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan], level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, 2, 3], level=0)
    mi = MultiIndex.from_tuples([(np.nan, 1), (1, 2)])
    msg = "labels \\['a'\\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, 'a'], level=0)