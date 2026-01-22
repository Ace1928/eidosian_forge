import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:Parquet format '2.0':FutureWarning")
def test_spark_flavor_preserves_pandas_metadata():
    df = _test_dataframe(size=100)
    df.index = np.arange(0, 10 * len(df), 10)
    df.index.name = 'foo'
    result = _roundtrip_pandas_dataframe(df, {'version': '2.0', 'flavor': 'spark'})
    tm.assert_frame_equal(result, df)