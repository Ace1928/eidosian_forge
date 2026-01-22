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
def test_metadata_schema_filesystem(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    fname = 'data.parquet'
    file_path = str(tempdir / fname)
    file_uri = 'file:///' + file_path
    pq.write_table(table, file_path)
    metadata = pq.read_metadata(tempdir / fname)
    schema = table.schema
    assert pq.read_metadata(file_uri).equals(metadata)
    assert pq.read_metadata(file_path, filesystem=LocalFileSystem()).equals(metadata)
    assert pq.read_metadata(fname, filesystem=f'file:///{tempdir}').equals(metadata)
    assert pq.read_schema(file_uri).equals(schema)
    assert pq.read_schema(file_path, filesystem=LocalFileSystem()).equals(schema)
    assert pq.read_schema(fname, filesystem=f'file:///{tempdir}').equals(schema)
    with util.change_cwd(tempdir):
        assert pq.read_metadata(fname, filesystem=LocalFileSystem()).equals(metadata)
        assert pq.read_schema(fname, filesystem=LocalFileSystem()).equals(schema)