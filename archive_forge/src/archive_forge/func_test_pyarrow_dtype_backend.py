import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_dtype_backend(self):
    pa = pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype=np.dtype('int32')), 'b': pd.Series(['x', 'y', None], dtype=np.dtype('O')), 'c': pd.Series([True, False, None], dtype=np.dtype('O')), 'd': pd.Series([np.nan, 100.5, 200], dtype=np.dtype('float')), 'e': pd.Series(pd.date_range('2022', periods=3)), 'f': pd.Series(pd.date_range('2022', periods=3, tz='UTC').as_unit('s')), 'g': pd.Series(pd.timedelta_range('1D', periods=3))})
    result = df.convert_dtypes(dtype_backend='pyarrow')
    expected = pd.DataFrame({'a': pd.arrays.ArrowExtensionArray(pa.array([1, 2, 3], type=pa.int32())), 'b': pd.arrays.ArrowExtensionArray(pa.array(['x', 'y', None])), 'c': pd.arrays.ArrowExtensionArray(pa.array([True, False, None])), 'd': pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])), 'e': pd.arrays.ArrowExtensionArray(pa.array([datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2), datetime.datetime(2022, 1, 3)], type=pa.timestamp(unit='ns'))), 'f': pd.arrays.ArrowExtensionArray(pa.array([datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2), datetime.datetime(2022, 1, 3)], type=pa.timestamp(unit='s', tz='UTC'))), 'g': pd.arrays.ArrowExtensionArray(pa.array([datetime.timedelta(1), datetime.timedelta(2), datetime.timedelta(3)], type=pa.duration('ns')))})
    tm.assert_frame_equal(result, expected)