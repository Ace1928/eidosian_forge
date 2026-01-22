import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_cython_agg_frame_columns():
    df = DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(level=0, axis='columns').mean()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(level=0, axis='columns').mean()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(level=0, axis='columns').mean()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(level=0, axis='columns').mean()