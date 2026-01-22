import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
@pytest.mark.parametrize('case', [IntIndex(5, np.array([1, 2], dtype=np.int32)), IntIndex(5, np.array([0, 2, 4], dtype=np.int32)), IntIndex(0, np.array([], dtype=np.int32)), IntIndex(5, np.array([], dtype=np.int32))])
def test_intersect_identical(self, case):
    assert case.intersect(case).equals(case)
    case = case.to_block_index()
    assert case.intersect(case).equals(case)