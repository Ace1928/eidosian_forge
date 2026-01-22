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
def test_read_table_within_decorator(self, make_csv_file, set_async_read_mode):

    @dummy_decorator()
    def wrapped_read_table(file, method):
        if method == 'pandas':
            return pandas.read_table(file)
        if method == 'modin':
            return pd.read_table(file)
    unique_filename = make_csv_file(delimiter='\t')
    pandas_df = wrapped_read_table(unique_filename, method='pandas')
    modin_df = wrapped_read_table(unique_filename, method='modin')
    if StorageFormat.get() == 'Hdk':
        modin_df, pandas_df = align_datetime_dtypes(modin_df, pandas_df)
    df_equals(modin_df, pandas_df)