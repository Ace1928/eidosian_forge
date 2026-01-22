import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('grouper', [lambda x: x, lambda x: x.groupby('A')], ids=['None', 'groupby'])
@pytest.mark.parametrize('method', ['mean', 'sum'])
def test_invalid_engine_kwargs(self, grouper, method):
    df = DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': range(4)})
    with pytest.raises(ValueError, match='cython engine does not'):
        getattr(grouper(df).ewm(com=1.0), method)(engine='cython', engine_kwargs={'nopython': True})