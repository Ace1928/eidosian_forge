import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_parquet_configuration_options(tempdir):
    size = 10000
    np.random.seed(0)
    df = pd.DataFrame({'uint8': np.arange(size, dtype=np.uint8), 'uint16': np.arange(size, dtype=np.uint16), 'uint32': np.arange(size, dtype=np.uint32), 'uint64': np.arange(size, dtype=np.uint64), 'int8': np.arange(size, dtype=np.int16), 'int16': np.arange(size, dtype=np.int16), 'int32': np.arange(size, dtype=np.int32), 'int64': np.arange(size, dtype=np.int64), 'float32': np.arange(size, dtype=np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0})
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    for use_dictionary in [True, False]:
        _write_table(arrow_table, filename, version='2.6', use_dictionary=use_dictionary)
        table_read = _read_table(filename)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)
    for write_statistics in [True, False]:
        _write_table(arrow_table, filename, version='2.6', write_statistics=write_statistics)
        table_read = _read_table(filename)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)
    for compression in ['NONE', 'SNAPPY', 'GZIP', 'LZ4', 'ZSTD']:
        if compression != 'NONE' and (not pa.lib.Codec.is_available(compression)):
            continue
        _write_table(arrow_table, filename, version='2.6', compression=compression)
        table_read = _read_table(filename)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)