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
def test_parquet_invalid_version(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    with pytest.raises(ValueError, match='Unsupported Parquet format version'):
        _write_table(table, tempdir / 'test_version.parquet', version='2.2')
    with pytest.raises(ValueError, match='Unsupported Parquet data page ' + 'version'):
        _write_table(table, tempdir / 'test_version.parquet', data_page_version='2.2')