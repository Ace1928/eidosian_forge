import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_write_to_dataset_pandas_preserve_extensiondtypes(tempdir):
    df = pd.DataFrame({'part': 'a', 'col': [1, 2, 3]})
    df['col'] = df['col'].astype('Int64')
    table = pa.table(df)
    pq.write_to_dataset(table, str(tempdir / 'case1'), partition_cols=['part'])
    result = pq.read_table(str(tempdir / 'case1')).to_pandas()
    tm.assert_frame_equal(result[['col']], df[['col']])
    pq.write_to_dataset(table, str(tempdir / 'case2'))
    result = pq.read_table(str(tempdir / 'case2')).to_pandas()
    tm.assert_frame_equal(result[['col']], df[['col']])
    pq.write_table(table, str(tempdir / 'data.parquet'))
    result = pq.read_table(str(tempdir / 'data.parquet')).to_pandas()
    tm.assert_frame_equal(result[['col']], df[['col']])