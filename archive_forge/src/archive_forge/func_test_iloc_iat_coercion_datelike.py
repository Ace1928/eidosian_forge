from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ser, expected', [[Series(['2014-01-01', '2014-02-02'], dtype='datetime64[ns]'), Timestamp('2014-02-02')], [Series(['1 days', '2 days'], dtype='timedelta64[ns]'), Timedelta('2 days')]])
def test_iloc_iat_coercion_datelike(self, indexer_ial, ser, expected):
    result = indexer_ial(ser)[1]
    assert result == expected