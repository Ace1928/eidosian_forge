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
def test_mv_rename(self):
    path = pjoin(self.tmp_path, 'mv-test')
    new_path = pjoin(self.tmp_path, 'mv-new-test')
    data = b'foobarbaz'
    with self.hdfs.open(path, 'wb') as f:
        f.write(data)
    assert self.hdfs.exists(path)
    self.hdfs.mv(path, new_path)
    assert not self.hdfs.exists(path)
    assert self.hdfs.exists(new_path)
    assert self.hdfs.cat(new_path) == data
    self.hdfs.rename(new_path, path)
    assert self.hdfs.cat(path) == data