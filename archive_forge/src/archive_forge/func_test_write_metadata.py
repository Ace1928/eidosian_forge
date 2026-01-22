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
@pytest.mark.filterwarnings('ignore:Parquet format:FutureWarning')
def test_write_metadata(tempdir):
    path = str(tempdir / 'metadata')
    schema = pa.schema([('a', 'int64'), ('b', 'float64')])
    pq.write_metadata(schema, path)
    parquet_meta = pq.read_metadata(path)
    schema_as_arrow = parquet_meta.schema.to_arrow_schema()
    assert schema_as_arrow.equals(schema)
    if schema_as_arrow.metadata:
        assert b'ARROW:schema' not in schema_as_arrow.metadata
    for version in ['1.0', '2.0', '2.4', '2.6']:
        pq.write_metadata(schema, path, version=version)
        parquet_meta = pq.read_metadata(path)
        expected_version = '1.0' if version == '1.0' else '2.6'
        assert parquet_meta.format_version == expected_version
    table = pa.table({'a': [1, 2], 'b': [0.1, 0.2]}, schema=schema)
    pq.write_table(table, tempdir / 'data.parquet')
    parquet_meta = pq.read_metadata(str(tempdir / 'data.parquet'))
    pq.write_metadata(schema, path, metadata_collector=[parquet_meta, parquet_meta])
    parquet_meta_mult = pq.read_metadata(path)
    assert parquet_meta_mult.num_row_groups == 2
    msg = 'AppendRowGroups requires equal schemas.\nThe two columns with index 0 differ.'
    with pytest.raises(RuntimeError, match=msg):
        pq.write_metadata(pa.schema([('a', 'int32'), ('b', 'null')]), path, metadata_collector=[parquet_meta, parquet_meta])