from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_at_with_tuple_index_get():
    df = DataFrame({'a': [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    assert df.at[(1, 2), 'a'] == 1
    series = df['a']
    assert series.index.nlevels == 1
    assert series.at[1, 2] == 1