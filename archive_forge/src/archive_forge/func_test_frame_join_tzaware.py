from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_frame_join_tzaware(self):
    test1 = DataFrame(np.zeros((6, 3)), index=date_range('2012-11-15 00:00:00', periods=6, freq='100ms', tz='US/Central'))
    test2 = DataFrame(np.zeros((3, 3)), index=date_range('2012-11-15 00:00:00', periods=3, freq='250ms', tz='US/Central'), columns=range(3, 6))
    result = test1.join(test2, how='outer')
    expected = test1.index.union(test2.index)
    tm.assert_index_equal(result.index, expected)
    assert result.index.tz.zone == 'US/Central'