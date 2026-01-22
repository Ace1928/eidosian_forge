import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_sort_special():
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    result = idx.difference([])
    tm.assert_index_equal(result, idx)