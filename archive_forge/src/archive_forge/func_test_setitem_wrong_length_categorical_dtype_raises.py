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
def test_setitem_wrong_length_categorical_dtype_raises(self):
    cat = Categorical.from_codes([0, 1, 1, 0, 1, 2], ['a', 'b', 'c'])
    df = DataFrame(range(10), columns=['bar'])
    msg = f'Length of values \\({len(cat)}\\) does not match length of index \\({len(df)}\\)'
    with pytest.raises(ValueError, match=msg):
        df['foo'] = cat