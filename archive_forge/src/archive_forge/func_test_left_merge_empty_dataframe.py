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
def test_left_merge_empty_dataframe(self):
    left = DataFrame({'key': [1], 'value': [2]})
    right = DataFrame({'key': []})
    result = merge(left, right, on='key', how='left')
    tm.assert_frame_equal(result, left)
    result = merge(right, left, on='key', how='right')
    tm.assert_frame_equal(result, left)