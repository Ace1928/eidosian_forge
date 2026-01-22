import pytest
from pandas import (
import pandas._testing as tm
def test_isetitem_dimension_mismatch(self):
    df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    value = df.copy()
    with pytest.raises(ValueError, match='Got 2 positions but value has 3 columns'):
        df.isetitem([1, 2], value)
    value = df.copy()
    with pytest.raises(ValueError, match='Got 2 positions but value has 1 columns'):
        df.isetitem([1, 2], value[['a']])