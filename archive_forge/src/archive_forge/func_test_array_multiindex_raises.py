import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_array_multiindex_raises():
    idx = pd.MultiIndex.from_product([['A'], ['a', 'b']])
    msg = 'MultiIndex has no single backing array'
    with pytest.raises(ValueError, match=msg):
        idx.array