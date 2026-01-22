import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_parquet_2_0_roundtrip_read_pandas_no_index_written(tempdir):
    df = alltypes_sample(size=10000)
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    js = arrow_table.schema.pandas_metadata
    assert not js['index_columns']
    assert js['columns']
    _write_table(arrow_table, filename)
    table_read = pq.read_pandas(filename)
    js = table_read.schema.pandas_metadata
    assert not js['index_columns']
    read_metadata = table_read.schema.metadata
    assert arrow_table.schema.metadata == read_metadata
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)