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
def test_getitem_boolean_indexing_mixed(self):
    df = DataFrame({0: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 1: {35: np.nan, 40: 0.326323168594462, 43: np.nan, 49: 0.326323168594462, 50: 0.3911472448057814}, 2: {35: np.nan, 40: np.nan, 43: 0.29012581014105987, 49: np.nan, 50: np.nan}, 3: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 4: {35: 0.34215328467153283, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 'y': {35: 0, 40: 0, 43: 0, 49: 0, 50: 1}})
    df2 = df.copy()
    df2[df2 > 0.3] = 1
    expected = df.copy()
    expected.loc[40, 1] = 1
    expected.loc[49, 1] = 1
    expected.loc[50, 1] = 1
    expected.loc[35, 4] = 1
    tm.assert_frame_equal(df2, expected)
    df['foo'] = 'test'
    msg = 'not supported between instances|unorderable types'
    with pytest.raises(TypeError, match=msg):
        df[df > 0.3] = 1