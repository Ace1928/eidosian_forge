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
def test_read_csv_glob_4373(self, tmp_path):
    columns, filename = (['col0'], str(tmp_path / '1x1.csv'))
    df = pd.DataFrame([[1]], columns=columns)
    with warns_that_defaulting_to_pandas() if Engine.get() == 'Dask' else contextlib.nullcontext():
        df.to_csv(filename)
    kwargs = {'filepath_or_buffer': filename, 'usecols': columns}
    modin_df = pd.read_csv_glob(**kwargs)
    pandas_df = pandas.read_csv(**kwargs)
    df_equals(modin_df, pandas_df)