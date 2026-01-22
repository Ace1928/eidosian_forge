from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('orient,item_getter', [('dict', lambda d, col, idx: d[col][idx]), ('records', lambda d, col, idx: d[idx][col]), ('list', lambda d, col, idx: d[col][idx]), ('split', lambda d, col, idx: d['data'][idx][d['columns'].index(col)]), ('index', lambda d, col, idx: d[idx][col])])
def test_to_dict_box_scalars(self, orient, item_getter):
    df = DataFrame({'a': [1, 2], 'b': [0.1, 0.2]})
    result = df.to_dict(orient=orient)
    assert isinstance(item_getter(result, 'a', 0), int)
    assert isinstance(item_getter(result, 'b', 0), float)