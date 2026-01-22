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
def test_precision_loss():
    df = DataFrame([[sum((2 ** i for i in range(60))), sum((2 ** i for i in range(52)))]], columns=['big', 'little'])
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(PossiblePrecisionLoss, match='Column converted from int64 to float64'):
            df.to_stata(path, write_index=False)
        reread = read_stata(path)
        expected_dt = Series([np.float64, np.float64], index=['big', 'little'])
        tm.assert_series_equal(reread.dtypes, expected_dt)
        assert reread.loc[0, 'little'] == df.loc[0, 'little']
        assert reread.loc[0, 'big'] == float(df.loc[0, 'big'])