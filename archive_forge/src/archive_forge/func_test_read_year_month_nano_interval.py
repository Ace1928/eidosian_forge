from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_read_year_month_nano_interval(tmpdir):
    """ARROW-15783: Verify to_pandas works for interval types.

    Interval types require static structures to be enabled. This test verifies
    that they are when no other library functions are invoked.
    """
    mdn_interval_type = pa.month_day_nano_interval()
    schema = pa.schema([pa.field('nums', mdn_interval_type)])
    path = tmpdir.join('file.arrow').strpath
    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            interval_array = pa.array([(1, 2, 3)], type=mdn_interval_type)
            batch = pa.record_batch([interval_array], schema)
            writer.write(batch)
    invoke_script('read_record_batch.py', path)