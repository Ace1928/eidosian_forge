from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_loc_boolean_indexer_miss_matching_index():
    ser = Series([1])
    indexer = Series([NA, False], dtype='boolean', index=[1, 2])
    with pytest.raises(IndexingError, match='Unalignable'):
        ser.loc[indexer]