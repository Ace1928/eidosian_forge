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
def test_ignore_hidden_files_dot(tempdir):
    dirpath = tempdir / guid()
    dirpath.mkdir()
    paths = _make_example_multifile_dataset(dirpath, nfiles=10, file_nrows=5)
    with (dirpath / '.DS_Store').open('wb') as f:
        f.write(b'gibberish')
    with (dirpath / '.private').open('wb') as f:
        f.write(b'gibberish')
    dataset = pq.ParquetDataset(dirpath)
    _assert_dataset_paths(dataset, paths)