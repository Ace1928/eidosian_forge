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
def test_read_filters(self, engine, tmp_path):
    df = pd.DataFrame({'int': list(range(4)), 'part': list('aabb')})
    expected = pd.DataFrame({'int': [0, 1]})
    check_round_trip(df, engine, path=tmp_path, expected=expected, write_kwargs={'partition_cols': ['part']}, read_kwargs={'filters': [('part', '==', 'a')], 'columns': ['int']}, repeat=1)