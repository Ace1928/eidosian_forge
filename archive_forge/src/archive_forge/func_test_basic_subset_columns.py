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
def test_basic_subset_columns(self, pa, df_full):
    df = df_full
    df['datetime_tz'] = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
    check_round_trip(df, pa, expected=df[['string', 'int']], read_kwargs={'columns': ['string', 'int']})