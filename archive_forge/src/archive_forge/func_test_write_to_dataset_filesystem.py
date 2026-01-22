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
def test_write_to_dataset_filesystem(tempdir):
    df = pd.DataFrame({'A': [1, 2, 3]})
    table = pa.Table.from_pandas(df)
    path = str(tempdir)
    pq.write_to_dataset(table, path, filesystem=fs.LocalFileSystem())
    result = pq.read_table(path)
    assert result.equals(table)