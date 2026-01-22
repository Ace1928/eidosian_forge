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
def test_parquet_file_page_index():
    for write_page_index in (False, True):
        table = pa.table({'a': [1, 2, 3]})
        writer = pa.BufferOutputStream()
        _write_table(table, writer, write_page_index=write_page_index)
        reader = pa.BufferReader(writer.getvalue())
        metadata = pq.read_metadata(reader)
        cc = metadata.row_group(0).column(0)
        assert cc.has_offset_index is write_page_index
        assert cc.has_column_index is write_page_index