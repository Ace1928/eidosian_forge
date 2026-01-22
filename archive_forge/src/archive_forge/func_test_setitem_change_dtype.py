import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_change_dtype(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    dft = frame.T
    s = dft['foo', 'two']
    dft['foo', 'two'] = s > s.median()
    tm.assert_series_equal(dft['foo', 'two'], s > s.median())
    reindexed = dft.reindex(columns=[('foo', 'two')])
    tm.assert_series_equal(reindexed['foo', 'two'], s > s.median())