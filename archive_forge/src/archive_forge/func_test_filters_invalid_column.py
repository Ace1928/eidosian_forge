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
def test_filters_invalid_column(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [['integers', integer_keys]]
    N = 5
    df = pd.DataFrame({'index': np.arange(N), 'integers': np.array(integer_keys, dtype='i4')}, columns=['index', 'integers'])
    _generate_partition_directories(fs, base_path, partition_spec, df)
    msg = 'No match for FieldRef.Name\\(non_existent_column\\)'
    with pytest.raises(ValueError, match=msg):
        pq.ParquetDataset(base_path, filesystem=fs, filters=[('non_existent_column', '<', 3)]).read()