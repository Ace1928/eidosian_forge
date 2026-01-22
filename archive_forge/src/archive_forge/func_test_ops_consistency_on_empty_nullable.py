from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('method', ['mean', 'var'])
@pytest.mark.parametrize('dtype', ['Float64', 'Int64', 'boolean'])
def test_ops_consistency_on_empty_nullable(self, method, dtype):
    eser = Series([], dtype=dtype)
    result = getattr(eser, method)()
    assert result is pd.NA
    nser = Series([np.nan], dtype=dtype)
    result = getattr(nser, method)()
    assert result is pd.NA