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
def test_getitem_keyerror_with_integer_index(self, any_int_numpy_dtype):
    dtype = any_int_numpy_dtype
    ser = Series(np.random.default_rng(2).standard_normal(6), index=Index([0, 0, 1, 1, 2, 2], dtype=dtype))
    with pytest.raises(KeyError, match='^5$'):
        ser[5]
    with pytest.raises(KeyError, match="^'c'$"):
        ser['c']
    ser = Series(np.random.default_rng(2).standard_normal(6), index=[2, 2, 0, 0, 1, 1])
    with pytest.raises(KeyError, match='^5$'):
        ser[5]
    with pytest.raises(KeyError, match="^'c'$"):
        ser['c']