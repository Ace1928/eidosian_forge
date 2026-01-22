import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx,labels', [(period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']), (date_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']), (pd.timedelta_range(start='1 day', periods=20), ['3 day', '30 day'])])
def test_loc_with_list_of_strings_representing_datetimes_missing_value(self, idx, labels):
    ser = Series(range(20), index=idx)
    df = DataFrame(range(20), index=idx)
    msg = 'not in index'
    with pytest.raises(KeyError, match=msg):
        ser.loc[labels]
    with pytest.raises(KeyError, match=msg):
        ser[labels]
    with pytest.raises(KeyError, match=msg):
        df.loc[labels]