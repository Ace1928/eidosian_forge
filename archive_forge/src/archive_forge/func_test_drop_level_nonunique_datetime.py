import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_level_nonunique_datetime(self):
    idx = Index([2, 3, 4, 4, 5], name='id')
    idxdt = pd.to_datetime(['2016-03-23 14:00', '2016-03-23 15:00', '2016-03-23 16:00', '2016-03-23 16:00', '2016-03-23 17:00'])
    df = DataFrame(np.arange(10).reshape(5, 2), columns=list('ab'), index=idx)
    df['tstamp'] = idxdt
    df = df.set_index('tstamp', append=True)
    ts = Timestamp('201603231600')
    assert df.index.is_unique is False
    result = df.drop(ts, level='tstamp')
    expected = df.loc[idx != 4]
    tm.assert_frame_equal(result, expected)