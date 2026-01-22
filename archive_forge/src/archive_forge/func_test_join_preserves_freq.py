from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('tz', [None, 'US/Pacific'])
def test_join_preserves_freq(self, tz):
    dti = date_range('2016-01-01', periods=10, tz=tz)
    result = dti[:5].join(dti[5:], how='outer')
    assert result.freq == dti.freq
    tm.assert_index_equal(result, dti)
    result = dti[:5].join(dti[6:], how='outer')
    assert result.freq is None
    expected = dti.delete(5)
    tm.assert_index_equal(result, expected)