import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('other', [Categorical(['b', 'a'], categories=['b', 'a', 'c']), Categorical(['b', 'a'], categories=['a', 'b', 'c']), Categorical(['a', 'a'], categories=['a']), Categorical(['b', 'b'], categories=['b'])])
def test_setitem_different_unordered_raises(self, other):
    target = Categorical(['a', 'b'], categories=['a', 'b'])
    mask = np.array([True, False])
    msg = 'Cannot set a Categorical with another, without identical categories'
    with pytest.raises(TypeError, match=msg):
        target[mask] = other[mask]