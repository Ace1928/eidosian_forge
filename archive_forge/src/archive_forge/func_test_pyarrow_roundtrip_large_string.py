from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.large_memory
def test_pyarrow_roundtrip_large_string():
    data = np.array([b'x' * 1024] * (3 * 1024 ** 2), dtype='object')
    arr = pa.array(data, type=pa.large_string())
    table = pa.table([arr], names=['large_string'])
    result = _from_dataframe(table.__dataframe__())
    col = result.__dataframe__().get_column(0)
    assert col.size() == 3 * 1024 ** 2
    assert pa.types.is_large_string(table[0].type)
    assert pa.types.is_large_string(result[0].type)
    assert table.equals(result)