import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_object_dtype(self):
    arr = SparseArray(['A', 'A', np.nan, 'B'], dtype=object)
    assert arr.dtype == SparseDtype(object)
    assert np.isnan(arr.fill_value)
    arr = SparseArray(['A', 'A', np.nan, 'B'], dtype=object, fill_value='A')
    assert arr.dtype == SparseDtype(object, 'A')
    assert arr.fill_value == 'A'