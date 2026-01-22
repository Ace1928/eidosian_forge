import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_filter_before_validate_schema(tempdir):
    dir1 = tempdir / 'A=0'
    dir1.mkdir()
    table1 = pa.Table.from_pandas(pd.DataFrame({'B': [1, 2, 3]}))
    pq.write_table(table1, dir1 / 'data.parquet')
    dir2 = tempdir / 'A=1'
    dir2.mkdir()
    table2 = pa.Table.from_pandas(pd.DataFrame({'B': ['a', 'b', 'c']}))
    pq.write_table(table2, dir2 / 'data.parquet')
    table = pq.read_table(tempdir, filters=[[('A', '==', 0)]])
    assert table.column('B').equals(pa.chunked_array([[1, 2, 3]]))