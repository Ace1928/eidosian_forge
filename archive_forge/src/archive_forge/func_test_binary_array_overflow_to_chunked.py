import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.slow
@pytest.mark.pandas
@pytest.mark.large_memory
def test_binary_array_overflow_to_chunked():
    values = [b'x'] + [b'x' * (1 << 20)] * 2 * (1 << 10)
    df = pd.DataFrame({'byte_col': values})
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    read_tbl = _simple_table_roundtrip(tbl)
    col0_data = read_tbl[0]
    assert isinstance(col0_data, pa.ChunkedArray)
    assert col0_data.num_chunks == 2
    assert tbl.equals(read_tbl)