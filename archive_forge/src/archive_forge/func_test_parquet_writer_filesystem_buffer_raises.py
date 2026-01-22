import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
def test_parquet_writer_filesystem_buffer_raises():
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    filesystem = fs.LocalFileSystem()
    with pytest.raises(ValueError, match='specified path is file-like'):
        pq.ParquetWriter(pa.BufferOutputStream(), table.schema, filesystem=filesystem)