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
def test_reads_over_batch(tempdir):
    data = [None] * (1 << 20)
    data.append([1])
    table = pa.Table.from_arrays([data], ['column'])
    path = tempdir / 'arrow-11607.parquet'
    pq.write_table(table, path)
    table2 = pq.read_table(path)
    assert table == table2