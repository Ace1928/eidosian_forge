import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'uint64', 'float64', 'complex128', 'period[M]', 'timedelta64', 'timedelta64[ns]', 'datetime64', 'datetime64[ns]', 'datetime64[ns, US/Eastern]'])
def test_astype_cannot_cast(self, index, dtype):
    msg = 'Cannot cast IntervalIndex to dtype'
    with pytest.raises(TypeError, match=msg):
        index.astype(dtype)