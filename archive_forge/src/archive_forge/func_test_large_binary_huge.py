import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.slow
@pytest.mark.large_memory
def test_large_binary_huge():
    s = b'xy' * 997
    data = [s] * ((1 << 33) // len(s))
    for type in [pa.large_binary(), pa.large_string()]:
        arr = pa.array(data, type=type)
        table = pa.Table.from_arrays([arr], names=['strs'])
        for use_dictionary in [False, True]:
            _check_roundtrip(table, use_dictionary=use_dictionary)
        del arr, table