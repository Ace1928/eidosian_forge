from datetime import datetime
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name, exp', [('b', 'b'), ('c', None)])
def test_append_names_match(name, exp):
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['a', 'b'])
    midx2 = MultiIndex.from_arrays([[3], [5]], names=['a', name])
    result = midx.append(midx2)
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=['a', exp])
    tm.assert_index_equal(result, expected)