from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dtype', ('m8[ns]', 'm8[ns]', 'M8[ns]', 'M8[ns, UTC]'))
@pytest.mark.parametrize('skipna', [True, False])
def test_empty_timeseries_reductions_return_nat(self, dtype, skipna):
    assert Series([], dtype=dtype).min(skipna=skipna) is NaT
    assert Series([], dtype=dtype).max(skipna=skipna) is NaT