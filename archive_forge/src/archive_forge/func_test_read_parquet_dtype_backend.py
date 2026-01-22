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
@pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
def test_read_parquet_dtype_backend(self, engine, make_parquet_file, dtype_backend):
    with ensure_clean('.parquet') as unique_filename:
        make_parquet_file(filename=unique_filename, row_group_size=100)

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        expected_exception = None
        if engine == 'fastparquet':
            expected_exception = ValueError("The 'dtype_backend' argument is not supported for the fastparquet engine")
        eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, dtype_backend=dtype_backend, comparator=comparator, expected_exception=expected_exception)