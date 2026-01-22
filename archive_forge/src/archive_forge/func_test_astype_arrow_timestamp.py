import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_astype_arrow_timestamp(using_copy_on_write):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [Timestamp('2020-01-01 01:01:01.000001'), Timestamp('2020-01-01 01:01:01.000001')]}, dtype='M8[ns]')
    result = df.astype('timestamp[ns][pyarrow]')
    if using_copy_on_write:
        assert not result._mgr._has_no_reference(0)
        if pa_version_under12p0:
            assert not np.shares_memory(get_array(df, 'a'), get_array(result, 'a')._pa_array)
        else:
            assert np.shares_memory(get_array(df, 'a'), get_array(result, 'a')._pa_array)