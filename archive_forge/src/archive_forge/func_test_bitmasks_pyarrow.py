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
@pytest.mark.parametrize(('offset', 'length', 'expected_values'), [(0, None, [3.3, float('nan'), 2.1]), (1, None, [float('nan'), 2.1]), (2, None, [2.1]), (0, 2, [3.3, float('nan')]), (0, 1, [3.3]), (1, 1, [float('nan')])])
def test_bitmasks_pyarrow(offset, length, expected_values):
    pa = pytest.importorskip('pyarrow', '11.0.0')
    arr = [3.3, None, 2.1]
    table = pa.table({'arr': arr}).slice(offset, length)
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    expected = pd.DataFrame({'arr': expected_values})
    tm.assert_frame_equal(result, expected)
    assert pa.Table.equals(pa.interchange.from_dataframe(result), table)