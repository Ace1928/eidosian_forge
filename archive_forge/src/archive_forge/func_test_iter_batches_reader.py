import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
@pytest.mark.parametrize('chunk_size', [1000])
def test_iter_batches_reader(tempdir, chunk_size):
    df = alltypes_sample(size=10000, categorical=True)
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert arrow_table.schema.pandas_metadata is not None
    _write_table(arrow_table, filename, version='2.6', chunk_size=chunk_size)
    file_ = pq.ParquetFile(filename)

    def get_all_batches(f):
        for row_group in range(f.num_row_groups):
            batches = f.iter_batches(batch_size=900, row_groups=[row_group])
            for batch in batches:
                yield batch
    batches = list(get_all_batches(file_))
    batch_no = 0
    for i in range(file_.num_row_groups):
        tm.assert_frame_equal(batches[batch_no].to_pandas(), file_.read_row_groups([i]).to_pandas().head(900))
        batch_no += 1
        tm.assert_frame_equal(batches[batch_no].to_pandas().reset_index(drop=True), file_.read_row_groups([i]).to_pandas().iloc[900:].reset_index(drop=True))
        batch_no += 1