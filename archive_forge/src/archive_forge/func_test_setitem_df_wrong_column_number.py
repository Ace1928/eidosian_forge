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
@pytest.mark.parametrize('cols', [['a', 'b', 'c'], ['a', 'a', 'a']])
def test_setitem_df_wrong_column_number(self, cols):
    df = DataFrame([[1, 2, 3]], columns=cols)
    rhs = DataFrame([[10, 11]], columns=['d', 'e'])
    msg = 'Columns must be same length as key'
    with pytest.raises(ValueError, match=msg):
        df['a'] = rhs