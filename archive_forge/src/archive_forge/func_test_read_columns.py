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
def test_read_columns(self, engine):
    df = pd.DataFrame({'string': list('abc'), 'int': list(range(1, 4))})
    expected = pd.DataFrame({'string': list('abc')})
    check_round_trip(df, engine, expected=expected, read_kwargs={'columns': ['string']})