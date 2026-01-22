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
def test_read_sql_with_chunksize(self, make_sql_connection):
    filename = get_unique_filename(extension='db')
    table = 'test_read_sql_with_chunksize'
    conn = make_sql_connection(filename, table)
    query = f'select * from {table}'
    pandas_gen = pandas.read_sql(query, conn, chunksize=10)
    modin_gen = pd.read_sql(query, conn, chunksize=10)
    for modin_df, pandas_df in zip(modin_gen, pandas_gen):
        df_equals(modin_df, pandas_df)