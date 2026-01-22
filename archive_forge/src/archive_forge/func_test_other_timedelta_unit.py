from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_other_timedelta_unit(self, unit):
    df1 = DataFrame({'entity_id': [101, 102]})
    ser = Series([None, None], index=[101, 102], name='days')
    dtype = f'm8[{unit}]'
    if unit in ['D', 'h', 'm']:
        msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)
        df2 = ser.astype('m8[s]').to_frame('days')
    else:
        df2 = ser.astype(dtype).to_frame('days')
        assert df2['days'].dtype == dtype
    result = df1.merge(df2, left_on='entity_id', right_index=True)
    exp = DataFrame({'entity_id': [101, 102], 'days': np.array(['nat', 'nat'], dtype=dtype)}, columns=['entity_id', 'days'])
    tm.assert_frame_equal(result, exp)