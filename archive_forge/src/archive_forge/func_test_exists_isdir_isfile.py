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
def test_exists_isdir_isfile(self):
    dir_path = pjoin(self.tmp_path, 'info-base')
    file_path = pjoin(dir_path, 'ex')
    missing_path = pjoin(dir_path, 'this-path-is-missing')
    self.hdfs.mkdir(dir_path)
    with self.hdfs.open(file_path, 'wb') as f:
        f.write(b'foobarbaz')
    assert self.hdfs.exists(dir_path)
    assert self.hdfs.exists(file_path)
    assert not self.hdfs.exists(missing_path)
    assert self.hdfs.isdir(dir_path)
    assert not self.hdfs.isdir(file_path)
    assert not self.hdfs.isdir(missing_path)
    assert not self.hdfs.isfile(dir_path)
    assert self.hdfs.isfile(file_path)
    assert not self.hdfs.isfile(missing_path)