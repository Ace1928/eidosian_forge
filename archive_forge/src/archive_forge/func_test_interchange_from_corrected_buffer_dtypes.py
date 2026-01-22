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
def test_interchange_from_corrected_buffer_dtypes(monkeypatch) -> None:
    df = pd.DataFrame({'a': ['foo', 'bar']}).__dataframe__()
    interchange = df.__dataframe__()
    column = interchange.get_column_by_name('a')
    buffers = column.get_buffers()
    buffers_data = buffers['data']
    buffer_dtype = buffers_data[1]
    buffer_dtype = (DtypeKind.UINT, 8, ArrowCTypes.UINT8, buffer_dtype[3])
    buffers['data'] = (buffers_data[0], buffer_dtype)
    column.get_buffers = lambda: buffers
    interchange.get_column_by_name = lambda _: column
    monkeypatch.setattr(df, '__dataframe__', lambda allow_copy: interchange)
    pd.api.interchange.from_dataframe(df)