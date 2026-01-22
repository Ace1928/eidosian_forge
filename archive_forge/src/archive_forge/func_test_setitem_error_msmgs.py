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
def test_setitem_error_msmgs(self):
    df = DataFrame({'bar': [1, 2, 3], 'baz': ['d', 'e', 'f']}, index=Index(['a', 'b', 'c'], name='foo'))
    ser = Series(['g', 'h', 'i', 'j'], index=Index(['a', 'b', 'c', 'a'], name='foo'), name='fiz')
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        df['newcol'] = ser
    df = DataFrame(np.random.default_rng(2).integers(0, 2, (4, 4)), columns=['a', 'b', 'c', 'd'])
    msg = 'Cannot set a DataFrame with multiple columns to the single column gr'
    with pytest.raises(ValueError, match=msg):
        df['gr'] = df.groupby(['b', 'c']).count()
    msg = 'Cannot set a DataFrame without columns to the column gr'
    with pytest.raises(ValueError, match=msg):
        df['gr'] = DataFrame()