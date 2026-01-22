import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('other', [Categorical(['b', 'a']), Categorical(['b', 'a'], categories=['b', 'a'])])
def test_setitem_same_but_unordered(self, other):
    target = Categorical(['a', 'b'], categories=['a', 'b'])
    mask = np.array([True, False])
    target[mask] = other[mask]
    expected = Categorical(['b', 'b'], categories=['a', 'b'])
    tm.assert_categorical_equal(target, expected)