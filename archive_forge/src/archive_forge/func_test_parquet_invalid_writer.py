import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
def test_parquet_invalid_writer(tempdir):
    with pytest.raises(TypeError):
        some_schema = pa.schema([pa.field('x', pa.int32())])
        pq.ParquetWriter(None, some_schema)
    with pytest.raises(TypeError):
        pq.ParquetWriter(tempdir / 'some_path', None)