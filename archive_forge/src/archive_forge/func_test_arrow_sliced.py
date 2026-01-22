import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def test_arrow_sliced(data):
    df = pd.DataFrame({'a': data})
    table = pa.table(df)
    result = table.slice(2, None).to_pandas()
    expected = df.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)
    df2 = df.fillna(data[0])
    table = pa.table(df2)
    result = table.slice(2, None).to_pandas()
    expected = df2.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)