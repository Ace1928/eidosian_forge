from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_integer_categorical_not_positional(self):
    ser = Series(['a', 'b', 'c'], index=Index([1, 2, 3], dtype='category'))
    assert ser.get(3) == 'c'
    assert ser[3] == 'c'