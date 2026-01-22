import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_merging_parquet_tables_with_different_pandas_metadata(tempdir):
    schema = pa.schema([pa.field('int', pa.int16()), pa.field('float', pa.float32()), pa.field('string', pa.string())])
    df1 = pd.DataFrame({'int': np.arange(3, dtype=np.uint8), 'float': np.arange(3, dtype=np.float32), 'string': ['ABBA', 'EDDA', 'ACDC']})
    df2 = pd.DataFrame({'int': [4, 5], 'float': [1.1, None], 'string': [None, None]})
    table1 = pa.Table.from_pandas(df1, schema=schema, preserve_index=False)
    table2 = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)
    assert not table1.schema.equals(table2.schema, check_metadata=True)
    assert table1.schema.equals(table2.schema)
    writer = pq.ParquetWriter(tempdir / 'merged.parquet', schema=schema)
    writer.write_table(table1)
    writer.write_table(table2)