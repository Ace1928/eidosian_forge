from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('values, dtype', [([1, 2, NA], 'Int64'), (['A', 'B', NA], 'string')])
@pytest.mark.parametrize('as_frame', [True, False])
def test_pickle_roundtrip_containers(as_frame, values, dtype):
    s = pd.Series(pd.array(values, dtype=dtype))
    if as_frame:
        s = s.to_frame(name='A')
    result = tm.round_trip_pickle(s)
    tm.assert_equal(result, s)