from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('func', [['min'], ['mean', 'max'], {'A': 'sum'}, {'A': 'prod', 'B': 'median'}])
def test_multi_agg_axis_1_raises(func):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    index.name = 'date'
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index).T
    warning_msg = 'DataFrame.resample with axis=1 is deprecated.'
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        res = df.resample('ME', axis=1)
        with pytest.raises(NotImplementedError, match='axis other than 0 is not supported'):
            res.agg(func)