import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_read_empty_dta_with_dtypes(self, version):
    empty_df_typed = DataFrame({'i8': np.array([0], dtype=np.int8), 'i16': np.array([0], dtype=np.int16), 'i32': np.array([0], dtype=np.int32), 'i64': np.array([0], dtype=np.int64), 'u8': np.array([0], dtype=np.uint8), 'u16': np.array([0], dtype=np.uint16), 'u32': np.array([0], dtype=np.uint32), 'u64': np.array([0], dtype=np.uint64), 'f32': np.array([0], dtype=np.float32), 'f64': np.array([0], dtype=np.float64)})
    expected = empty_df_typed.copy()
    expected['u8'] = expected['u8'].astype(np.int8)
    expected['u16'] = expected['u16'].astype(np.int16)
    expected['u32'] = expected['u32'].astype(np.int32)
    expected['u64'] = expected['u64'].astype(np.int32)
    expected['i64'] = expected['i64'].astype(np.int32)
    with tm.ensure_clean() as path:
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread = read_stata(path)
        tm.assert_frame_equal(expected, empty_reread)
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)