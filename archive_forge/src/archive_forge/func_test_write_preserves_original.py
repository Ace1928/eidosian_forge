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
def test_write_preserves_original(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=list('abcd'))
    df.loc[2, 'a':'c'] = np.nan
    df_copy = df.copy()
    with tm.ensure_clean() as path:
        df.to_stata(path, write_index=False)
    tm.assert_frame_equal(df, df_copy)