from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_interchange_from_non_pandas_tz_aware(request):
    pa = pytest.importorskip('pyarrow', '11.0.0')
    import pyarrow.compute as pc
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(raises=pa.ArrowInvalid, reason='TODO: Set ARROW_TIMEZONE_DATABASE environment variable on CI to path to the tzdata for pyarrow.')
        request.applymarker(mark)
    arr = pa.array([datetime(2020, 1, 1), None, datetime(2020, 1, 2)])
    arr = pc.assume_timezone(arr, 'Asia/Kathmandu')
    table = pa.table({'arr': arr})
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    expected = pd.DataFrame(['2020-01-01 00:00:00+05:45', 'NaT', '2020-01-02 00:00:00+05:45'], columns=['arr'], dtype='datetime64[us, Asia/Kathmandu]')
    tm.assert_frame_equal(expected, result)