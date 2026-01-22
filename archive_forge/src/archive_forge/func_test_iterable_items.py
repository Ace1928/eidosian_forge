import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dtype, rdtype', dtypes)
def test_iterable_items(self, dtype, rdtype):
    s = Series([1], dtype=dtype)
    _, result = next(iter(s.items()))
    assert isinstance(result, rdtype)
    _, result = next(iter(s.items()))
    assert isinstance(result, rdtype)