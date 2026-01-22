from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tar_suffix', ['.tar', '.tar.gz'])
def test_read_tarfile(c_parser_only, csv_dir_path, tar_suffix):
    parser = c_parser_only
    tar_path = os.path.join(csv_dir_path, 'tar_csv' + tar_suffix)
    with tarfile.open(tar_path, 'r') as tar:
        data_file = tar.extractfile('tar_data.csv')
        out = parser.read_csv(data_file)
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(out, expected)