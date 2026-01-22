from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_timestamp(self):
    tsmp = Timestamp('20130101')
    test_data = DataFrame({'A': [tsmp, tsmp], 'B': [tsmp, tsmp]})
    test_data_mixed = DataFrame({'A': [tsmp, tsmp], 'B': [1, 2]})
    expected_records = [{'A': tsmp, 'B': tsmp}, {'A': tsmp, 'B': tsmp}]
    expected_records_mixed = [{'A': tsmp, 'B': 1}, {'A': tsmp, 'B': 2}]
    assert test_data.to_dict(orient='records') == expected_records
    assert test_data_mixed.to_dict(orient='records') == expected_records_mixed
    expected_series = {'A': Series([tsmp, tsmp], name='A'), 'B': Series([tsmp, tsmp], name='B')}
    expected_series_mixed = {'A': Series([tsmp, tsmp], name='A'), 'B': Series([1, 2], name='B')}
    tm.assert_dict_equal(test_data.to_dict(orient='series'), expected_series)
    tm.assert_dict_equal(test_data_mixed.to_dict(orient='series'), expected_series_mixed)
    expected_split = {'index': [0, 1], 'data': [[tsmp, tsmp], [tsmp, tsmp]], 'columns': ['A', 'B']}
    expected_split_mixed = {'index': [0, 1], 'data': [[tsmp, 1], [tsmp, 2]], 'columns': ['A', 'B']}
    tm.assert_dict_equal(test_data.to_dict(orient='split'), expected_split)
    tm.assert_dict_equal(test_data_mixed.to_dict(orient='split'), expected_split_mixed)