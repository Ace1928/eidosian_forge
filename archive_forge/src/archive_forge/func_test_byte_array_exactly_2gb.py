import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.slow
@pytest.mark.large_memory
def test_byte_array_exactly_2gb():
    val = b'x' * (1 << 10)
    base = pa.array([val] * ((1 << 21) - 1))
    cases = [[b'x' * 1023], [b'x' * 1024], [b'x' * 1025]]
    for case in cases:
        values = pa.chunked_array([base, pa.array(case)])
        t = pa.table([values], names=['f0'])
        result = _simple_table_roundtrip(t, use_dictionary=False)
        assert t.equals(result)