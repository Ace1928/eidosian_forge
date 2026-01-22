from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_timeseries_repr_object_dtype(self):
    index = Index([datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], dtype=object)
    ts = Series(np.random.randn(len(index)), index)
    repr(ts)
    ts = tm.makeTimeSeries(1000)
    assert repr(ts).splitlines()[-1].startswith('Freq:')
    ts2 = ts.iloc[np.random.randint(0, len(ts) - 1, 400)]
    repr(ts2).splitlines()[-1]