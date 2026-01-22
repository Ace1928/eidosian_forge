from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.fixture
def s_main_dtypes():
    """
    A DataFrame with many dtypes

    * datetime
    * datetimetz
    * timedelta
    * [u]int{8,16,32,64}
    * float{32,64}

    The columns are the name of the dtype.
    """
    df = pd.DataFrame({'datetime': pd.to_datetime(['2003', '2002', '2001', '2002', '2005']), 'datetimetz': pd.to_datetime(['2003', '2002', '2001', '2002', '2005']).tz_localize('US/Eastern'), 'timedelta': pd.to_timedelta(['3d', '2d', '1d', '2d', '5d'])})
    for dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64']:
        df[dtype] = Series([3, 2, 1, 2, 5], dtype=dtype)
    return df