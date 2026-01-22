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
def test_metadata_equals():
    table = pa.table({'a': [1, 2, 3]})
    with pa.BufferOutputStream() as out:
        pq.write_table(table, out)
        buf = out.getvalue()
    original_metadata = pq.read_metadata(pa.BufferReader(buf))
    match = "Argument 'other' has incorrect type"
    with pytest.raises(TypeError, match=match):
        original_metadata.equals(None)