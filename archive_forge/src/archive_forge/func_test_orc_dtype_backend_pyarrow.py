import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
def test_orc_dtype_backend_pyarrow():
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'string': list('abc'), 'string_with_nan': ['a', np.nan, 'c'], 'string_with_none': ['a', None, 'c'], 'bytes': [b'foo', b'bar', None], 'int': list(range(1, 4)), 'float': np.arange(4.0, 7.0, dtype='float64'), 'float_with_nan': [2.0, np.nan, 3.0], 'bool': [True, False, True], 'bool_with_na': [True, False, None], 'datetime': pd.date_range('20130101', periods=3), 'datetime_with_nat': [pd.Timestamp('20130101'), pd.NaT, pd.Timestamp('20130103')]})
    bytes_data = df.copy().to_orc()
    result = read_orc(BytesIO(bytes_data), dtype_backend='pyarrow')
    expected = pd.DataFrame({col: pd.arrays.ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns})
    tm.assert_frame_equal(result, expected)