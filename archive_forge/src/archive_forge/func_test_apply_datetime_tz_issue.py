from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_datetime_tz_issue(engine, request):
    if engine == 'numba':
        mark = pytest.mark.xfail(reason="numba engine doesn't support non-numeric indexes")
        request.node.add_marker(mark)
    timestamps = [Timestamp('2019-03-15 12:34:31.909000+0000', tz='UTC'), Timestamp('2019-03-15 12:34:34.359000+0000', tz='UTC'), Timestamp('2019-03-15 12:34:34.660000+0000', tz='UTC')]
    df = DataFrame(data=[0, 1, 2], index=timestamps)
    result = df.apply(lambda x: x.name, axis=1, engine=engine)
    expected = Series(index=timestamps, data=timestamps)
    tm.assert_series_equal(result, expected)