import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
@pytest.mark.parametrize('arr', [pa.nulls(10), pa.chunked_array([pa.nulls(4), pa.nulls(6)])])
def test_from_arrow_null(data, arr):
    res = data.dtype.__from_arrow__(arr)
    assert res.isna().all()
    assert len(res) == 10