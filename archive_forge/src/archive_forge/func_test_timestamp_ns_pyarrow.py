from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_timestamp_ns_pyarrow():
    pytest.importorskip('pyarrow', '11.0.0')
    timestamp_args = {'year': 2000, 'month': 1, 'day': 1, 'hour': 1, 'minute': 1, 'second': 1}
    df = pd.Series([datetime(**timestamp_args)], dtype='timestamp[ns][pyarrow]', name='col0').to_frame()
    dfi = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(dfi)['col0'].item()
    expected = pd.Timestamp(**timestamp_args)
    assert result == expected