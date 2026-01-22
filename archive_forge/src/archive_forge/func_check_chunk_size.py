import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
def check_chunk_size(data_size, chunk_size, expect_num_chunks):
    table = pa.Table.from_arrays([_range_integers(data_size, 'b')], names=['x'])
    if chunk_size is None:
        pq.write_table(table, tempdir / 'test.parquet')
    else:
        pq.write_table(table, tempdir / 'test.parquet', row_group_size=chunk_size)
    metadata = pq.read_metadata(tempdir / 'test.parquet')
    expected_chunk_size = default_chunk_size if chunk_size is None else chunk_size
    assert metadata.num_row_groups == expect_num_chunks
    latched_chunk_size = min(expected_chunk_size, abs_max_chunk_size)
    for chunk_idx in range(expect_num_chunks - 1):
        assert metadata.row_group(chunk_idx).num_rows == latched_chunk_size
    remainder = data_size - expected_chunk_size * (expect_num_chunks - 1)
    if remainder == 0:
        assert metadata.row_group(expect_num_chunks - 1).num_rows == latched_chunk_size
    else:
        assert metadata.row_group(expect_num_chunks - 1).num_rows == remainder