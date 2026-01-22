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
def test_read_parquet_list_of_files_5698(self, engine, make_parquet_file):
    if engine == 'fastparquet' and os.name == 'nt':
        pytest.xfail(reason='https://github.com/pandas-dev/pandas/issues/51720')
    with ensure_clean('.parquet') as f1, ensure_clean('.parquet') as f2, ensure_clean('.parquet') as f3:
        for f in [f1, f2, f3]:
            make_parquet_file(filename=f)
        eval_io(fn_name='read_parquet', path=[f1, f2, f3], engine=engine)