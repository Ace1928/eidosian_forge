import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('other', (Index([1, 2], dtype=np.int64), Index([1.0, 2.0], dtype=object), Index([1, 2], dtype=object)))
def test_equals_numeric_other_index_type(self, other):
    idx = Index([1.0, 2.0])
    assert idx.equals(other)
    assert other.equals(idx)