import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_itertuples_tuple_name(self):
    df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    tup = next(df.itertuples(name='TestName'))
    assert tup._fields == ('Index', 'a', 'b')
    assert (tup.Index, tup.a, tup.b) == tup
    assert type(tup).__name__ == 'TestName'