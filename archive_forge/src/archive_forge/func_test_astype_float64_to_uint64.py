import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_float64_to_uint64(self):
    idx = Index([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
    result = idx.astype('u8')
    expected = Index([0, 5, 10, 15, 20], dtype=np.uint64)
    tm.assert_index_equal(result, expected, exact=True)
    idx_with_negatives = idx - 10
    with pytest.raises(ValueError, match='losslessly'):
        idx_with_negatives.astype(np.uint64)