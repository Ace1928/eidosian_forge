import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_mixed_fixed(tmp_path, setup_path):
    complex64 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j], dtype=np.complex64)
    complex128 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j], dtype=np.complex128)
    df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd'], 'C': complex64, 'D': complex128, 'E': [1.0, 2.0, 3.0, 4.0]}, index=list('abcd'))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df')
    reread = read_hdf(path, 'df')
    tm.assert_frame_equal(df, reread)