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
def test_loc_iloc_setitem_non_categorical_rhs(self, orig, exp_parts_cats_col, indexer):
    df = orig.copy()
    key = (slice(2, 4), 0)
    if indexer is tm.loc:
        key = (slice('j', 'k'), df.columns[0])
    indexer(df)[key] = ['b', 'b']
    tm.assert_frame_equal(df, exp_parts_cats_col)
    with pytest.raises(TypeError, match=msg1):
        indexer(df)[key] = ['c', 'c']