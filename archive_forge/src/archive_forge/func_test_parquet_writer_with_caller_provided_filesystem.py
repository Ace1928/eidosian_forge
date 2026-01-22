import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
def test_parquet_writer_with_caller_provided_filesystem():
    out = pa.BufferOutputStream()

    class CustomFS(FileSystem):

        def __init__(self):
            self.path = None
            self.mode = None

        def open(self, path, mode='rb'):
            self.path = path
            self.mode = mode
            return out
    fs = CustomFS()
    fname = 'expected_fname.parquet'
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pq.ParquetWriter(fname, table.schema, filesystem=fs, version='2.6') as writer:
        writer.write_table(table)
    assert fs.path == fname
    assert fs.mode == 'wb'
    assert out.closed
    buf = out.getvalue()
    table_read = _read_table(pa.BufferReader(buf))
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df_read, df)
    with pytest.raises(ValueError) as err_info:
        pq.ParquetWriter(pa.BufferOutputStream(), table.schema, filesystem=fs)
        expected_msg = 'filesystem passed but where is file-like, so there is nothing to open with filesystem.'
        assert str(err_info) == expected_msg