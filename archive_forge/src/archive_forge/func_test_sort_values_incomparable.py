import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_sort_values_incomparable():
    mi = MultiIndex.from_arrays([[1, Timestamp('2000-01-01')], [3, 4]])
    match = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=match):
        mi.sort_values()