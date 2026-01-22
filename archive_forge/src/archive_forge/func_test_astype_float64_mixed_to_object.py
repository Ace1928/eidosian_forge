import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_float64_mixed_to_object(self):
    idx = Index([1.5, 2, 3, 4, 5], dtype=np.float64)
    idx.name = 'foo'
    result = idx.astype(object)
    assert result.equals(idx)
    assert idx.equals(result)
    assert isinstance(result, Index) and result.dtype == object