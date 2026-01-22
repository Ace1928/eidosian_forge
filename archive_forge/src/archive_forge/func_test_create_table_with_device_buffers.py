import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_create_table_with_device_buffers():
    hbuf, htable, dbuf, dtable = make_table_cuda()
    dtable2 = pa.Table.from_arrays(dtable.columns, dtable.column_names)
    assert htable.schema == dtable2.schema
    assert htable.num_rows == dtable2.num_rows
    assert htable.num_columns == dtable2.num_columns
    assert hbuf.equals(dbuf.copy_to_host())
    assert htable.equals(pa.ipc.open_stream(dbuf.copy_to_host()).read_all())