import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_parquet_nested_convenience(tempdir):
    df = pd.DataFrame({'a': [[1, 2, 3], None, [4, 5], []], 'b': [[1.0], None, None, [6.0, 7.0]]})
    path = str(tempdir / 'nested_convenience.parquet')
    table = pa.Table.from_pandas(df, preserve_index=False)
    _write_table(table, path)
    read = pq.read_table(path, columns=['a'])
    tm.assert_frame_equal(read.to_pandas(), df[['a']])
    read = pq.read_table(path, columns=['a', 'b'])
    tm.assert_frame_equal(read.to_pandas(), df)