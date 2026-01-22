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
def test_intelligently_handle_join_key(self):
    left = DataFrame({'key': [1, 1, 2, 2, 3], 'value': list(range(5))}, columns=['value', 'key'])
    right = DataFrame({'key': [1, 1, 2, 3, 4, 5], 'rvalue': list(range(6))})
    joined = merge(left, right, on='key', how='outer')
    expected = DataFrame({'key': [1, 1, 1, 1, 2, 2, 3, 4, 5], 'value': np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]), 'rvalue': [0, 1, 0, 1, 2, 2, 3, 4, 5]}, columns=['value', 'key', 'rvalue'])
    tm.assert_frame_equal(joined, expected)