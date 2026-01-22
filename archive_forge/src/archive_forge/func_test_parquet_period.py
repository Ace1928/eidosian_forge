import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.parquet
def test_parquet_period(tmpdir, registered_period_type):
    period_type, period_class = registered_period_type
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(period_type, storage)
    table = pa.table([arr], names=['ext'])
    import pyarrow.parquet as pq
    filename = tmpdir / 'period_extension_type.parquet'
    pq.write_table(table, filename)
    meta = pq.read_metadata(filename)
    assert meta.schema.column(0).physical_type == 'INT64'
    assert b'ARROW:schema' in meta.metadata
    import base64
    decoded_schema = base64.b64decode(meta.metadata[b'ARROW:schema'])
    schema = pa.ipc.read_schema(pa.BufferReader(decoded_schema))
    assert schema.field('ext').metadata == {}
    result = pq.read_table(filename)
    result.validate(full=True)
    assert result.schema.field('ext').type == period_type
    assert result.schema.field('ext').metadata == {}
    result_array = result.column('ext').chunk(0)
    assert type(result_array) is period_class
    pa.unregister_extension_type(period_type.extension_name)
    result = pq.read_table(filename)
    result.validate(full=True)
    assert result.schema.field('ext').type == pa.int64()
    assert result.schema.field('ext').metadata == {b'ARROW:extension:metadata': b'freq=D', b'ARROW:extension:name': b'test.period'}