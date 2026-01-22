import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.pandas
def test_read_schema(tempdir):
    N = 100
    df = pd.DataFrame({'index': np.arange(N), 'values': np.random.randn(N)}, columns=['index', 'values'])
    data_path = tempdir / 'test.parquet'
    table = pa.Table.from_pandas(df)
    _write_table(table, data_path)
    read1 = pq.read_schema(data_path)
    read2 = pq.read_schema(data_path, memory_map=True)
    assert table.schema.equals(read1)
    assert table.schema.equals(read2)
    assert table.schema.metadata[b'pandas'] == read1.metadata[b'pandas']