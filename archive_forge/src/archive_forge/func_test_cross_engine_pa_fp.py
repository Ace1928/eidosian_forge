import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
def test_cross_engine_pa_fp(df_cross_compat, pa, fp):
    df = df_cross_compat
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=pa, compression=None)
        result = read_parquet(path, engine=fp)
        tm.assert_frame_equal(result, df)
        result = read_parquet(path, engine=fp, columns=['a', 'd'])
        tm.assert_frame_equal(result, df[['a', 'd']])