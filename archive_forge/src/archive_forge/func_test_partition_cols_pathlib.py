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
@pytest.mark.parametrize('path_type', [str, lambda x: x], ids=['string', 'pathlib.Path'])
def test_partition_cols_pathlib(self, tmp_path, pa, df_compat, path_type):
    partition_cols = 'B'
    partition_cols_list = [partition_cols]
    df = df_compat
    path = path_type(tmp_path)
    df.to_parquet(path, partition_cols=partition_cols_list)
    assert read_parquet(path).shape == df.shape