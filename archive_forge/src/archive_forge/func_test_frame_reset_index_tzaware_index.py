from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
def test_frame_reset_index_tzaware_index(self, tz):
    dr = date_range('2012-06-02', periods=10, tz=tz)
    df = DataFrame(np.random.default_rng(2).standard_normal(len(dr)), dr)
    roundtripped = df.reset_index().set_index('index')
    xp = df.index.tz
    rs = roundtripped.index.tz
    assert xp == rs