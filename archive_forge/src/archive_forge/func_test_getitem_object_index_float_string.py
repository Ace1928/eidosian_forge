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
def test_getitem_object_index_float_string(self):
    ser = Series([1] * 4, index=Index(['a', 'b', 'c', 1.0]))
    assert ser['a'] == 1
    assert ser[1.0] == 1