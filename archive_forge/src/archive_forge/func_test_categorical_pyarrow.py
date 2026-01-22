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
def test_categorical_pyarrow():
    pa = pytest.importorskip('pyarrow', '11.0.0')
    arr = ['Mon', 'Tue', 'Mon', 'Wed', 'Mon', 'Thu', 'Fri', 'Sat', 'Sun']
    table = pa.table({'weekday': pa.array(arr).dictionary_encode()})
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    weekday = pd.Categorical(arr, categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    expected = pd.DataFrame({'weekday': weekday})
    tm.assert_frame_equal(result, expected)