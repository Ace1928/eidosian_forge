import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_object_multiindex(self):
    data_dic = {(0, datetime.date(2018, 3, 3)): {'A': 1, 'B': 10}, (0, datetime.date(2018, 3, 4)): {'A': 2, 'B': 11}, (1, datetime.date(2018, 3, 3)): {'A': 3, 'B': 12}, (1, datetime.date(2018, 3, 4)): {'A': 4, 'B': 13}}
    result = DataFrame.from_dict(data_dic, orient='index')
    data = {'A': [1, 2, 3, 4], 'B': [10, 11, 12, 13]}
    index = [[0, 0, 1, 1], [datetime.date(2018, 3, 3), datetime.date(2018, 3, 4), datetime.date(2018, 3, 3), datetime.date(2018, 3, 4)]]
    expected = DataFrame(data=data, index=index)
    tm.assert_frame_equal(result, expected)