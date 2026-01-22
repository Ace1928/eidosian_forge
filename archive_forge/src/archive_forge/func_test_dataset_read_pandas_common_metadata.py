import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
@pytest.mark.parametrize('preserve_index', [True, False, None])
@pytest.mark.parametrize('metadata_fname', ['_metadata', '_common_metadata'])
def test_dataset_read_pandas_common_metadata(tempdir, preserve_index, metadata_fname):
    nfiles = 5
    size = 5
    dirpath = tempdir / guid()
    dirpath.mkdir()
    test_data = []
    frames = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)
        df.index = pd.Index(np.arange(i * size, (i + 1) * size, dtype='int64'), name='index')
        path = dirpath / '{}.parquet'.format(i)
        table = pa.Table.from_pandas(df, preserve_index=preserve_index)
        table = table.replace_schema_metadata(None)
        assert table.schema.metadata is None
        _write_table(table, path)
        test_data.append(table)
        frames.append(df)
        paths.append(path)
    table_for_metadata = pa.Table.from_pandas(df, preserve_index=preserve_index)
    pq.write_metadata(table_for_metadata.schema, dirpath / metadata_fname)
    dataset = pq.ParquetDataset(dirpath)
    columns = ['uint8', 'strings']
    result = dataset.read_pandas(columns=columns).to_pandas()
    expected = pd.concat([x[columns] for x in frames])
    expected.index.name = df.index.name if preserve_index is not False else None
    tm.assert_frame_equal(result, expected)