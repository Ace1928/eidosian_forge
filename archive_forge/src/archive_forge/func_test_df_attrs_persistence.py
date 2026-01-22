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
def test_df_attrs_persistence(self, tmp_path, pa):
    path = tmp_path / 'test_df_metadata.p'
    df = pd.DataFrame(data={1: [1]})
    df.attrs = {'test_attribute': 1}
    df.to_parquet(path, engine=pa)
    new_df = read_parquet(path, engine=pa)
    assert new_df.attrs == df.attrs