from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('int_vals, float_vals, exp_vals', [([1, 2, 3], [1.0, 2.0, 3.0], {'X': [1, 2, 3], 'Y': [1.0, 2.0, 3.0]}), ([1, 2, 3], [1.0, 3.0], {'X': [1, 3], 'Y': [1.0, 3.0]}), ([1, 2], [1.0, 2.0, 3.0], {'X': [1, 2], 'Y': [1.0, 2.0]})])
def test_merge_on_ints_floats(self, int_vals, float_vals, exp_vals):
    A = DataFrame({'X': int_vals})
    B = DataFrame({'Y': float_vals})
    expected = DataFrame(exp_vals)
    result = A.merge(B, left_on='X', right_on='Y')
    tm.assert_frame_equal(result, expected)
    result = B.merge(A, left_on='Y', right_on='X')
    tm.assert_frame_equal(result, expected[['Y', 'X']])