import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def test_read_whole_file(self):
    path = pjoin(self.tmp_path, 'read-whole-file')
    data = b'foo' * 1000
    with self.hdfs.open(path, 'wb') as f:
        f.write(data)
    with self.hdfs.open(path, 'rb') as f:
        result = f.read()
    assert result == data