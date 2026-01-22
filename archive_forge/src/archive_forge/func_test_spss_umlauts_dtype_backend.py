import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
def test_spss_umlauts_dtype_backend(datapath, dtype_backend):
    fname = datapath('io', 'data', 'spss', 'umlauts.sav')
    df = pd.read_spss(fname, convert_categoricals=False, dtype_backend=dtype_backend)
    expected = pd.DataFrame({'var1': [1.0, 2.0, 1.0, 3.0]}, dtype='Int64')
    if dtype_backend == 'pyarrow':
        pa = pytest.importorskip('pyarrow')
        from pandas.arrays import ArrowExtensionArray
        expected = pd.DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
    tm.assert_frame_equal(df, expected)