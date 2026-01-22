import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.pandas
@pytest.mark.parametrize(('data', 'type', 'physical_type', 'min_value', 'max_value', 'null_count', 'num_values', 'distinct_count'), [([1, 2, 2, None, 4], pa.uint8(), 'INT32', 1, 4, 1, 4, None), ([1, 2, 2, None, 4], pa.uint16(), 'INT32', 1, 4, 1, 4, None), ([1, 2, 2, None, 4], pa.uint32(), 'INT32', 1, 4, 1, 4, None), ([1, 2, 2, None, 4], pa.uint64(), 'INT64', 1, 4, 1, 4, None), ([-1, 2, 2, None, 4], pa.int8(), 'INT32', -1, 4, 1, 4, None), ([-1, 2, 2, None, 4], pa.int16(), 'INT32', -1, 4, 1, 4, None), ([-1, 2, 2, None, 4], pa.int32(), 'INT32', -1, 4, 1, 4, None), ([-1, 2, 2, None, 4], pa.int64(), 'INT64', -1, 4, 1, 4, None), ([-1.1, 2.2, 2.3, None, 4.4], pa.float32(), 'FLOAT', -1.1, 4.4, 1, 4, None), ([-1.1, 2.2, 2.3, None, 4.4], pa.float64(), 'DOUBLE', -1.1, 4.4, 1, 4, None), (['', 'b', chr(1000), None, 'aaa'], pa.binary(), 'BYTE_ARRAY', b'', chr(1000).encode('utf-8'), 1, 4, None), ([True, False, False, True, True], pa.bool_(), 'BOOLEAN', False, True, 0, 5, None), ([b'\x00', b'b', b'12', None, b'aaa'], pa.binary(), 'BYTE_ARRAY', b'\x00', b'b', 1, 4, None)])
def test_parquet_column_statistics_api(data, type, physical_type, min_value, max_value, null_count, num_values, distinct_count):
    df = pd.DataFrame({'data': data})
    schema = pa.schema([pa.field('data', type)])
    table = pa.Table.from_pandas(df, schema=schema, safe=False)
    fileh = make_sample_file(table)
    meta = fileh.metadata
    rg_meta = meta.row_group(0)
    col_meta = rg_meta.column(0)
    stat = col_meta.statistics
    assert stat.has_min_max
    assert _close(type, stat.min, min_value)
    assert _close(type, stat.max, max_value)
    assert stat.null_count == null_count
    assert stat.num_values == num_values
    assert stat.distinct_count == distinct_count
    assert stat.physical_type == physical_type