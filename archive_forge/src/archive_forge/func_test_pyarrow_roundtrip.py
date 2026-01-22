from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.parametrize('uint', [pa.uint8(), pa.uint16(), pa.uint32()])
@pytest.mark.parametrize('int', [pa.int8(), pa.int16(), pa.int32(), pa.int64()])
@pytest.mark.parametrize('float, np_float', [(pa.float16(), np.float16), (pa.float32(), np.float32), (pa.float64(), np.float64)])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('tz', ['America/New_York', '+07:30', '-04:30'])
@pytest.mark.parametrize('offset, length', [(0, 3), (0, 2), (1, 2), (2, 1)])
def test_pyarrow_roundtrip(uint, int, float, np_float, unit, tz, offset, length):
    from datetime import datetime as dt
    arr = [1, 2, None]
    dt_arr = [dt(2007, 7, 13), None, dt(2007, 7, 15)]
    table = pa.table({'a': pa.array(arr, type=uint), 'b': pa.array(arr, type=int), 'c': pa.array(np.array(arr, dtype=np_float), type=float, from_pandas=True), 'd': [True, False, True], 'e': [True, False, None], 'f': ['a', None, 'c'], 'g': pa.array(dt_arr, type=pa.timestamp(unit, tz=tz))})
    table = table.slice(offset, length)
    result = _from_dataframe(table.__dataframe__())
    assert table.equals(result)
    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()
    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()