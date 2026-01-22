from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_index_equal_empty_iterable():
    a = MultiIndex(levels=[[], []], codes=[[], []], names=['a', 'b'])
    b = MultiIndex.from_arrays(arrays=[[], []], names=['a', 'b'])
    tm.assert_index_equal(a, b)