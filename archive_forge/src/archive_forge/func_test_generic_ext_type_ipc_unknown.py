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
def test_generic_ext_type_ipc_unknown(registered_period_type):
    period_type, _ = registered_period_type
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(period_type, storage)
    batch = pa.RecordBatch.from_arrays([arr], ['ext'])
    buf = ipc_write_batch(batch)
    del batch
    pa.unregister_extension_type('test.period')
    batch = ipc_read_batch(buf)
    result = batch.column(0)
    assert isinstance(result, pa.Int64Array)
    ext_field = batch.schema.field('ext')
    assert ext_field.metadata == {b'ARROW:extension:metadata': b'freq=D', b'ARROW:extension:name': b'test.period'}