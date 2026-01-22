import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
@pytest.mark.parametrize('pre_buffer', [False, True])
def test_pre_buffer(pre_buffer):
    N, K = (10000, 4)
    df = alltypes_sample(size=N)
    a_table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(a_table, buf, row_group_size=N / K, compression='snappy', version='2.6')
    buf.seek(0)
    pf = pq.ParquetFile(buf, pre_buffer=pre_buffer)
    assert pf.read().num_rows == N