from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_tuples_empty():
    result = MultiIndex.from_tuples([], names=['a', 'b'])
    expected = MultiIndex.from_arrays(arrays=[[], []], names=['a', 'b'])
    tm.assert_index_equal(result, expected)