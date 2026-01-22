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
def test_ignore_hidden_files_underscore(tempdir):
    dirpath = tempdir / guid()
    dirpath.mkdir()
    paths = _make_example_multifile_dataset(dirpath, nfiles=10, file_nrows=5)
    with (dirpath / '_committed_123').open('wb') as f:
        f.write(b'abcd')
    with (dirpath / '_started_321').open('wb') as f:
        f.write(b'abcd')
    dataset = pq.ParquetDataset(dirpath)
    _assert_dataset_paths(dataset, paths)