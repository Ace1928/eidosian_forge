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
def test_empty_table_roundtrip():
    df = alltypes_sample(size=10)
    table = pa.Table.from_pandas(df)
    table = pa.Table.from_arrays([col.chunk(0)[:0] for col in table.itercolumns()], names=table.schema.names)
    assert table.schema.field('null').type == pa.null()
    assert table.schema.field('null_list').type == pa.list_(pa.null())
    _check_roundtrip(table, version='2.6')