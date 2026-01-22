import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_dtype_bool_conversion(self):
    df = DataFrame(data=np.random.default_rng(2).standard_normal((100, 50)))
    df = df.where(df > 0)
    bools = df > 0
    mask = isna(df)
    expected = bools.astype(object).mask(mask)
    result = bools.mask(mask)
    tm.assert_frame_equal(result, expected)