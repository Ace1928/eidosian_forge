import dateutil.tz
import numpy as np
import pytest
from pandas import (
from pandas.core.arrays import datetimes
@pytest.mark.parametrize('offset', [-5, -1, 0, 1])
def test_iteration_over_chunksize(self, offset, monkeypatch):
    chunksize = 5
    index = date_range('2000-01-01 00:00:00', periods=chunksize - offset, freq='min')
    num = 0
    with monkeypatch.context() as m:
        m.setattr(datetimes, '_ITER_CHUNKSIZE', chunksize)
        for stamp in index:
            assert index[num] == stamp
            num += 1
    assert num == len(index)