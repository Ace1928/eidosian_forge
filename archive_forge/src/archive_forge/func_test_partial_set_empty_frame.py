import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame(self):
    df = DataFrame()
    msg = 'cannot set a frame with no defined columns'
    with pytest.raises(ValueError, match=msg):
        df.loc[1] = 1
    with pytest.raises(ValueError, match=msg):
        df.loc[1] = Series([1], index=['foo'])
    msg = 'cannot set a frame with no defined index and a scalar'
    with pytest.raises(ValueError, match=msg):
        df.loc[:, 1] = 1