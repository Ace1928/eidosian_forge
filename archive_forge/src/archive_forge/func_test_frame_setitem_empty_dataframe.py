from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_frame_setitem_empty_dataframe(self):
    dti = DatetimeIndex(['2000-01-01'], dtype='M8[ns]', name='date')
    df = DataFrame({'date': dti}).set_index('date')
    df = df[0:0].copy()
    df['3010'] = None
    df['2010'] = None
    expected = DataFrame([], columns=['3010', '2010'], index=dti[:0])
    tm.assert_frame_equal(df, expected)