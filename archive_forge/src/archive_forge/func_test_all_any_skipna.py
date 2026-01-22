from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_all_any_skipna(self):
    s1 = Series([np.nan, True])
    s2 = Series([np.nan, False])
    assert s1.all(skipna=False)
    assert s1.all(skipna=True)
    assert s2.any(skipna=False)
    assert not s2.any(skipna=True)