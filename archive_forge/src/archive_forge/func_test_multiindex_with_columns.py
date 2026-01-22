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
def test_multiindex_with_columns(self, pa):
    engine = pa
    dates = pd.date_range('01-Jan-2018', '01-Dec-2018', freq='MS')
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((2 * len(dates), 3)), columns=list('ABC'))
    index1 = pd.MultiIndex.from_product([['Level1', 'Level2'], dates], names=['level', 'date'])
    index2 = index1.copy(names=None)
    for index in [index1, index2]:
        df.index = index
        check_round_trip(df, engine)
        check_round_trip(df, engine, read_kwargs={'columns': ['A', 'B']}, expected=df[['A', 'B']])