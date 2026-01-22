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
def check_error_on_write(self, df, engine, exc, err_msg):
    with tm.ensure_clean() as path:
        with pytest.raises(exc, match=err_msg):
            to_parquet(df, path, engine, compression=None)