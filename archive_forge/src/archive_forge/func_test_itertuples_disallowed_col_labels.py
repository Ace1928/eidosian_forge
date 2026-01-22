import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_itertuples_disallowed_col_labels(self):
    df = DataFrame(data={'def': [1, 2, 3], 'return': [4, 5, 6]})
    tup2 = next(df.itertuples(name='TestName'))
    assert tup2 == (0, 1, 4)
    assert tup2._fields == ('Index', '_1', '_2')