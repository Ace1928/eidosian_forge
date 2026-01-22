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
@pytest.mark.slow
def test_file_with_over_int16_max_row_groups():
    t = pa.table([list(range(40000))], names=['f0'])
    _check_roundtrip(t, row_group_size=1)