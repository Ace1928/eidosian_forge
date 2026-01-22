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
@pytest.mark.network
@pytest.mark.single_cpu
def test_parquet_read_from_url(self, httpserver, datapath, df_compat, engine):
    if engine != 'auto':
        pytest.importorskip(engine)
    with open(datapath('io', 'data', 'parquet', 'simple.parquet'), mode='rb') as f:
        httpserver.serve_content(content=f.read())
        df = read_parquet(httpserver.url)
    tm.assert_frame_equal(df, df_compat)