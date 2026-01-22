import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def test_arrow_roundtrip(data):
    df = pd.DataFrame({'a': data})
    table = pa.table(df)
    assert table.field('a').type == str(data.dtype.numpy_dtype)
    result = table.to_pandas()
    assert result['a'].dtype == data.dtype
    tm.assert_frame_equal(result, df)