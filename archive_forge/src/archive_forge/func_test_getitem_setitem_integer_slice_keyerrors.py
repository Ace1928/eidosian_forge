from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_setitem_integer_slice_keyerrors(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), index=range(0, 20, 2))
    cp = df.copy()
    cp.iloc[4:10] = 0
    assert (cp.iloc[4:10] == 0).values.all()
    cp = df.copy()
    cp.iloc[3:11] = 0
    assert (cp.iloc[3:11] == 0).values.all()
    result = df.iloc[2:6]
    result2 = df.loc[3:11]
    expected = df.reindex([4, 6, 8, 10])
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
    df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
    with pytest.raises(KeyError, match='^3$'):
        df2.loc[3:11]
    with pytest.raises(KeyError, match='^3$'):
        df2.loc[3:11] = 0