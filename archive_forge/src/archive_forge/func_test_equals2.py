from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals2(self):
    idx = TimedeltaIndex(['1 days', '2 days', 'NaT'])
    assert idx.equals(idx)
    assert idx.equals(idx.copy())
    assert idx.equals(idx.astype(object))
    assert idx.astype(object).equals(idx)
    assert idx.astype(object).equals(idx.astype(object))
    assert not idx.equals(list(idx))
    assert not idx.equals(pd.Series(idx))
    idx2 = TimedeltaIndex(['2 days', '1 days', 'NaT'])
    assert not idx.equals(idx2)
    assert not idx.equals(idx2.copy())
    assert not idx.equals(idx2.astype(object))
    assert not idx.astype(object).equals(idx2)
    assert not idx.astype(object).equals(idx2.astype(object))
    assert not idx.equals(list(idx2))
    assert not idx.equals(pd.Series(idx2))
    oob = Index([timedelta(days=10 ** 6)] * 3, dtype=object)
    assert not idx.equals(oob)
    assert not idx2.equals(oob)
    oob2 = Index([np.timedelta64(x) for x in oob], dtype=object)
    assert (oob == oob2).all()
    assert not idx.equals(oob2)
    assert not idx2.equals(oob2)
    oob3 = oob.map(np.timedelta64)
    assert (oob3 == oob).all()
    assert not idx.equals(oob3)
    assert not idx2.equals(oob3)