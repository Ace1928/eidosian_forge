import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_dups(self):
    df = concat([DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B']), DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])], axis=1)
    expected = concat([df, df], axis=1)
    result = df.join(df, rsuffix='_2')
    result.columns = expected.columns
    tm.assert_frame_equal(result, expected)
    w = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
    x = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
    y = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
    z = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
    dta = x.merge(y, left_index=True, right_index=True).merge(z, left_index=True, right_index=True, how='outer')
    with pytest.raises(pd.errors.MergeError, match="Passing 'suffixes' which cause duplicate columns"):
        dta.merge(w, left_index=True, right_index=True)