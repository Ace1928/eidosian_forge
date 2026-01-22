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
def test_read_partitioned_directory_s3fs(s3_example_s3fs):
    fs, path = s3_example_s3fs
    _partition_test_for_filesystem(fs, path)