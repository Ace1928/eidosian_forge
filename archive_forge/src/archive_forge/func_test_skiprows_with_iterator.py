from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_skiprows_with_iterator():
    data = '0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n    '
    df_iter = read_fwf(StringIO(data), colspecs=[(0, 2)], names=['a'], iterator=True, chunksize=2, skiprows=[0, 1, 2, 6, 9])
    expected_frames = [DataFrame({'a': [3, 4]}), DataFrame({'a': [5, 7]}, index=[2, 3]), DataFrame({'a': [8]}, index=[4])]
    for i, result in enumerate(df_iter):
        tm.assert_frame_equal(result, expected_frames[i])