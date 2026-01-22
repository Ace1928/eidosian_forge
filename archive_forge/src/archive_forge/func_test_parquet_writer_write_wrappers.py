import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
@pytest.mark.parametrize('filesystem', [None, LocalFileSystem._get_instance(), fs.LocalFileSystem()])
def test_parquet_writer_write_wrappers(tempdir, filesystem):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    path_table = str(tempdir / 'data_table.parquet')
    path_batch = str(tempdir / 'data_batch.parquet')
    with pq.ParquetWriter(path_table, table.schema, filesystem=filesystem, version='2.6') as writer:
        writer.write_table(table)
    result = _read_table(path_table).to_pandas()
    tm.assert_frame_equal(result, df)
    with pq.ParquetWriter(path_batch, table.schema, filesystem=filesystem, version='2.6') as writer:
        writer.write_batch(batch)
    result = _read_table(path_batch).to_pandas()
    tm.assert_frame_equal(result, df)
    with pq.ParquetWriter(path_table, table.schema, filesystem=filesystem, version='2.6') as writer:
        writer.write(table)
    result = _read_table(path_table).to_pandas()
    tm.assert_frame_equal(result, df)
    with pq.ParquetWriter(path_batch, table.schema, filesystem=filesystem, version='2.6') as writer:
        writer.write(batch)
    result = _read_table(path_batch).to_pandas()
    tm.assert_frame_equal(result, df)