import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('mode', ['wb', 'w'])
def test_to_csv_binary_handle(self, mode):
    """
        Binary file objects should work (if 'mode' contains a 'b') or even without
        it in most cases.

        GH 35058 and GH 19827
        """
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD')), index=Index([f'i-{i}' for i in range(30)]))
    with tm.ensure_clean() as path:
        with open(path, mode='w+b') as handle:
            df.to_csv(handle, mode=mode)
        tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))