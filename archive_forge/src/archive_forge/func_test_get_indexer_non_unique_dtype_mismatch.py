import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_non_unique_dtype_mismatch(self):
    indexes, missing = Index(['A', 'B']).get_indexer_non_unique(Index([0]))
    tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)
    tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)