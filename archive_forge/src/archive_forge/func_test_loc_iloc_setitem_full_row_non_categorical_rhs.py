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
@pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
def test_loc_iloc_setitem_full_row_non_categorical_rhs(self, orig, exp_single_row, indexer):
    df = orig.copy()
    key = 2
    if indexer is tm.loc:
        key = df.index[2]
    indexer(df)[key, :] = ['b', 2]
    tm.assert_frame_equal(df, exp_single_row)
    with pytest.raises(TypeError, match=msg1):
        indexer(df)[key, :] = ['c', 2]