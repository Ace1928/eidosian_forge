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
@pytest.mark.parametrize('parse_dates', [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()])
def test_read_single_csv_with_parse_dates(self, parse_dates):
    try:
        pandas_df = pandas.read_csv(time_parsing_csv_path, parse_dates=parse_dates)
    except Exception as pandas_exception:
        with pytest.raises(Exception) as modin_exception:
            modin_df = pd.read_csv_glob(time_parsing_csv_path, parse_dates=parse_dates)
            try_cast_to_pandas(modin_df)
        assert isinstance(modin_exception.value, type(pandas_exception)), 'Got Modin Exception type {}, but pandas Exception type {} was expected'.format(type(modin_exception.value), type(pandas_exception))
    else:
        modin_df = pd.read_csv_glob(time_parsing_csv_path, parse_dates=parse_dates)
        df_equals(modin_df, pandas_df)