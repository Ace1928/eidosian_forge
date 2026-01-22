import pytest
from pandas import (
import pandas._testing as tm
def test_append_mismatched_categories(self, ci):
    msg = 'all inputs must be Index'
    with pytest.raises(TypeError, match=msg):
        ci.append(ci.values.set_categories(list('abcd')))
    with pytest.raises(TypeError, match=msg):
        ci.append(ci.values.reorder_categories(list('abc')))