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
def test_additional_extension_arrays(self, pa):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype='Int64'), 'b': pd.Series([1, 2, 3], dtype='UInt32'), 'c': pd.Series(['a', None, 'c'], dtype='string')})
    check_round_trip(df, pa)
    df = pd.DataFrame({'a': pd.Series([1, 2, 3, None], dtype='Int64')})
    check_round_trip(df, pa)