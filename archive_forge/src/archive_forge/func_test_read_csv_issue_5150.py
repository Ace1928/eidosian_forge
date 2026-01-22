import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.parametrize('set_async_read_mode', [False, True], indirect=True)
def test_read_csv_issue_5150(self, set_async_read_mode):
    with ensure_clean('.csv') as unique_filename:
        pandas_df = pandas.DataFrame(np.random.randint(0, 100, size=(2 ** 6, 2 ** 6)))
        pandas_df.to_csv(unique_filename, index=False)
        expected_pandas_df = pandas.read_csv(unique_filename, index_col=False)
        modin_df = pd.read_csv(unique_filename, index_col=False)
        actual_pandas_df = modin_df._to_pandas()
        if AsyncReadMode.get():
            df_equals(expected_pandas_df, actual_pandas_df)
    if not AsyncReadMode.get():
        df_equals(expected_pandas_df, actual_pandas_df)