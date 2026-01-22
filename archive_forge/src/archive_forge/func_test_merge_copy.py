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
def test_merge_copy(self):
    left = DataFrame({'a': 0, 'b': 1}, index=range(10))
    right = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
    merged = merge(left, right, left_index=True, right_index=True, copy=True)
    merged['a'] = 6
    assert (left['a'] == 0).all()
    merged['d'] = 'peekaboo'
    assert (right['d'] == 'bar').all()