import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left,right', [(RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)), (RangeIndex(0), RangeIndex(1, -1, 3)), (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)), (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2))])
def test_equals_range(self, left, right):
    assert left.equals(right)
    assert right.equals(left)