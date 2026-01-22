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
def test_write_to_dataset_no_partitions_s3fs(s3_example_s3fs):
    fs, path = s3_example_s3fs
    _test_write_to_dataset_no_partitions(path, filesystem=fs)