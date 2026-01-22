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
@pytest.mark.parametrize('read_mode', ['r', 'rb'])
@pytest.mark.parametrize('buffer_start_pos', [0, 10])
@pytest.mark.parametrize('set_async_read_mode', [False, True], indirect=True)
def test_read_csv_file_handle(self, read_mode, make_csv_file, buffer_start_pos, set_async_read_mode):
    unique_filename = make_csv_file()
    with open(unique_filename, mode=read_mode) as buffer:
        buffer.seek(buffer_start_pos)
        pandas_df = pandas.read_csv(buffer)
        buffer.seek(buffer_start_pos)
        modin_df = pd.read_csv(buffer)
    df_equals(modin_df, pandas_df)