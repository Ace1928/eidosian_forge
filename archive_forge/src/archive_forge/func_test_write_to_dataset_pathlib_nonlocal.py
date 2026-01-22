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
@pytest.mark.s3
def test_write_to_dataset_pathlib_nonlocal(tempdir, s3_example_s3fs):
    fs, _ = s3_example_s3fs
    with pytest.raises(TypeError, match='path-like objects are only allowed'):
        _test_write_to_dataset_with_partitions(tempdir / 'test1', filesystem=fs)
    with pytest.raises(TypeError, match='path-like objects are only allowed'):
        _test_write_to_dataset_no_partitions(tempdir / 'test2', filesystem=fs)