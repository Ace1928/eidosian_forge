import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
@pytest.mark.parametrize('nrows', [35, 100])
def test_read_multiple_csv_nrows(self, request, nrows):
    pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
    pandas_df = pandas_df.iloc[:nrows, :]
    modin_df = pd.read_csv_glob(pytest.glob_path, nrows=nrows)
    pandas_df = pandas_df.reset_index(drop=True)
    modin_df = modin_df.reset_index(drop=True)
    df_equals(modin_df, pandas_df)