from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_empty_row_groups(tempdir):
    table = pa.Table.from_arrays([pa.array([], type='int32')], ['f0'])
    path = tempdir / 'empty_row_groups.parquet'
    num_groups = 3
    with pq.ParquetWriter(path, table.schema) as writer:
        for i in range(num_groups):
            writer.write_table(table)
    reader = pq.ParquetFile(path)
    assert reader.metadata.num_row_groups == num_groups
    for i in range(num_groups):
        assert reader.read_row_group(i).equals(table)