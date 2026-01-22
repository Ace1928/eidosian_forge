import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_across_dimensions(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1j, 1.0 + 1j, 1.0 + 1j, 1.0 + 1j])
    s = Series(complex128, index=list('abcd'))
    df = DataFrame({'A': s, 'B': s})
    path = tmp_path / setup_path
    df.to_hdf(path, key='obj', format='table')
    reread = read_hdf(path, 'obj')
    tm.assert_frame_equal(df, reread)