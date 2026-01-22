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
@pytest.mark.parametrize('indexer, idx', [(tm.loc, 1), (tm.iloc, 2)])
def test_setitem_value_coercing_dtypes(self, indexer, idx):
    df = DataFrame([['1', np.nan], ['2', np.nan], ['3', np.nan]], dtype=object)
    rhs = DataFrame([[1, np.nan], [2, np.nan]])
    indexer(df)[:idx, :] = rhs
    expected = DataFrame([[1, np.nan], [2, np.nan], ['3', np.nan]], dtype=object)
    tm.assert_frame_equal(df, expected)