import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.pandas
def test_extension_to_pandas_storage_type(registered_period_type):
    period_type, _ = registered_period_type
    np_arr = np.array([1, 2, 3, 4], dtype='i8')
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(period_type, storage)
    if isinstance(period_type, PeriodTypeWithToPandasDtype):
        pandas_dtype = period_type.to_pandas_dtype()
    else:
        pandas_dtype = np_arr.dtype
    result = arr.to_pandas()
    assert result.dtype == pandas_dtype
    chunked_arr = pa.chunked_array([arr])
    result = chunked_arr.to_numpy()
    assert result.dtype == np_arr.dtype
    result = chunked_arr.to_pandas()
    assert result.dtype == pandas_dtype
    data = [pa.array([1, 2, 3, 4]), pa.array(['foo', 'bar', None, None]), pa.array([True, None, True, False]), arr]
    my_schema = pa.schema([('f0', pa.int8()), ('f1', pa.string()), ('f2', pa.bool_()), ('ext', period_type)])
    table = pa.Table.from_arrays(data, schema=my_schema)
    result = table.to_pandas()
    assert result['ext'].dtype == pandas_dtype
    import pandas as pd
    if Version(pd.__version__) >= Version('2.1.0'):
        result = table.to_pandas(types_mapper=pd.ArrowDtype)
        assert isinstance(result['ext'].dtype, pd.ArrowDtype)