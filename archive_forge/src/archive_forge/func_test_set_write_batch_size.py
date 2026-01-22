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
@pytest.mark.pandas
def test_set_write_batch_size():
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    _check_roundtrip(table, data_page_size=10, write_batch_size=1, version='2.4')