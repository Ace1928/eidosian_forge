from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient, expected', [('split', {'columns': ['col1', 'col2'], 'data': [[1, 3], [2, 4]]}), ('tight', {'columns': ['col1', 'col2'], 'data': [[1, 3], [2, 4]], 'column_names': [None]})])
def test_to_dict_index_false(self, orient, expected):
    df = DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['row1', 'row2'])
    result = df.to_dict(orient=orient, index=False)
    tm.assert_dict_equal(result, expected)