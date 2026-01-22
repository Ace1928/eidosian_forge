import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_read_pandas_passthrough_keywords(tempdir):
    df = pd.DataFrame({'a': [1, 2, 3]})
    filename = tempdir / 'data.parquet'
    _write_table(df, filename)
    result = pq.read_pandas('data.parquet', filesystem=SubTreeFileSystem(str(tempdir), LocalFileSystem()))
    assert result.equals(pa.table(df))