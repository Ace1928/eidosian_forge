from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_series_repr_ordered(self):
    s = Series(Categorical([1, 2, 3], ordered=True))
    exp = '0    1\n1    2\n2    3\ndtype: category\nCategories (3, int64): [1 < 2 < 3]'
    assert repr(s) == exp
    s = Series(Categorical(np.arange(10), ordered=True))
    exp = f'0    0\n1    1\n2    2\n3    3\n4    4\n5    5\n6    6\n7    7\n8    8\n9    9\ndtype: category\nCategories (10, {np.int_().dtype}): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]'
    assert repr(s) == exp