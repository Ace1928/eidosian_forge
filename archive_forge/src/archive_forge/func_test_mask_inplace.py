import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_inplace(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    cond = df > 0
    rdf = df.copy()
    return_value = rdf.where(cond, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(rdf, df.where(cond))
    tm.assert_frame_equal(rdf, df.mask(~cond))
    rdf = df.copy()
    return_value = rdf.where(cond, -df, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(rdf, df.where(cond, -df))
    tm.assert_frame_equal(rdf, df.mask(~cond, -df))