import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dtype_inference_overflows():
    idx = Index(np.array([1, 2, 3], dtype=np.int8))
    result = idx.map(lambda x: x * 1000)
    expected = Index([1000, 2000, 3000], dtype=np.int64)
    tm.assert_index_equal(result, expected)