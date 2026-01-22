from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_intersection_duplicates_all_indexes(index):
    if index.empty:
        pytest.skip('Not relevant for empty Index')
    idx = index
    idx_non_unique = idx[[0, 0, 1, 2]]
    assert idx.intersection(idx_non_unique).equals(idx_non_unique.intersection(idx))
    assert idx.intersection(idx_non_unique).is_unique