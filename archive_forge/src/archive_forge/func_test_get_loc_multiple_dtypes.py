from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype1', [int, float, bool, str])
@pytest.mark.parametrize('dtype2', [int, float, bool, str])
def test_get_loc_multiple_dtypes(self, dtype1, dtype2):
    levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
    idx = MultiIndex.from_product(levels)
    assert idx.get_loc(idx[2]) == 2