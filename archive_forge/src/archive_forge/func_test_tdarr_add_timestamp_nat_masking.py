from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('str_ts', ['1950-01-01', '1980-01-01'])
def test_tdarr_add_timestamp_nat_masking(self, box_with_array, str_ts):
    tdinat = pd.to_timedelta(['24658 days 11:15:00', 'NaT'])
    tdobj = tm.box_expected(tdinat, box_with_array)
    ts = Timestamp(str_ts)
    ts_variants = [ts, ts.to_pydatetime(), ts.to_datetime64().astype('datetime64[ns]'), ts.to_datetime64().astype('datetime64[D]')]
    for variant in ts_variants:
        res = tdobj + variant
        if box_with_array is DataFrame:
            assert res.iloc[1, 1] is NaT
        else:
            assert res[1] is NaT