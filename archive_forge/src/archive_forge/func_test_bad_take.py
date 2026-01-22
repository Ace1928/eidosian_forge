import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_bad_take(self, arr):
    with pytest.raises(IndexError, match='bounds'):
        arr.take([11])