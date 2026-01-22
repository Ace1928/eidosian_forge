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
@pytest.mark.parametrize('indexer', [['a'], 'a'])
@pytest.mark.parametrize('col', [{}, {'b': 1}])
def test_set_2d_casting_date_to_int(self, col, indexer):
    df = DataFrame({'a': [Timestamp('2022-12-29'), Timestamp('2022-12-30')], **col})
    df.loc[[1], indexer] = df['a'] + pd.Timedelta(days=1)
    expected = DataFrame({'a': [Timestamp('2022-12-29'), Timestamp('2022-12-31')], **col})
    tm.assert_frame_equal(df, expected)