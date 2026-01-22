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
@pytest.mark.xfail(reason='legacy.FileSystem not supported with ParquetDataset due to legacy path being removed in PyArrow 15.0.0.', raises=TypeError)
@pytest.mark.pandas
@pytest.mark.parquet
def test_read_multiple_parquet_files(self):
    tmpdir = pjoin(self.tmp_path, 'multi-parquet-' + guid())
    self.hdfs.mkdir(tmpdir)
    expected = self._write_multiple_hdfs_pq_files(tmpdir)
    result = self.hdfs.read_parquet(tmpdir)
    assert_frame_equal(result.to_pandas().sort_values(by='index').reset_index(drop=True), expected.to_pandas())