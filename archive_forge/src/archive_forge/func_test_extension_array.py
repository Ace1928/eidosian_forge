from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.parametrize('arg, exp', [[[1, 3, 2], [1, 2, 3]], [[1, 3, np.nan, 2], [1, 2, 3, np.nan]]])
def test_extension_array(self, arg, exp):
    a = array(arg, dtype='Int64')
    result = safe_sort(a)
    expected = array(exp, dtype='Int64')
    tm.assert_extension_array_equal(result, expected)