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
def test_parquet_compression_roundtrip(tempdir):
    table = pa.table([pa.array(range(4))], names=['ints'])
    path = tempdir / 'arrow-10480.pyarrow.gz'
    pq.write_table(table, path, compression='GZIP')
    result = pq.read_table(path)
    assert result.equals(table)