import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_is_copy(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
    df2 = df.sample(3)
    with tm.assert_produces_warning(None):
        df2['d'] = 1