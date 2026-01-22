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
@pytest.mark.parametrize('filters', [None, [], [('B', '==', 'a')], [('B', '==', 'a'), ('A', '>=', 50000), ('idx', '<=', 30000), ('idx_categorical', '==', 'y')]])
def test_read_parquet_pandas_index(self, engine, filters):
    if version.parse(pa.__version__) >= version.parse('12.0.0') and version.parse(pd.__version__) < version.parse('2.0.0') and (engine == 'pyarrow'):
        pytest.xfail('incompatible versions; see #6072')
    pandas_df = pandas.DataFrame({'idx': np.random.randint(0, 100000, size=2000), 'idx_categorical': pandas.Categorical(['y', 'z'] * 1000), 'idx_periodrange': pandas.period_range(start='2017-01-01', periods=2000), 'A': np.random.randint(0, 100000, size=2000), 'B': ['a', 'b'] * 1000, 'C': ['c'] * 2000})
    if version.parse(pa.__version__) >= version.parse('8.0.0'):
        pandas_df['idx_timedelta'] = pandas.timedelta_range(start='1 day', periods=2000)
    if engine == 'pyarrow':
        pandas_df['idx_datetime'] = pandas.date_range(start='1/1/2018', periods=2000)
    for col in pandas_df.columns:
        if col.startswith('idx'):
            if col == 'idx_categorical' and engine == 'fastparquet' and (version.parse(fastparquet.__version__) < version.parse('2023.1.0')):
                continue
            with ensure_clean('.parquet') as unique_filename:
                pandas_df.set_index(col).to_parquet(unique_filename)
                eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters)
    with ensure_clean('.parquet') as unique_filename:
        pandas_df.set_index(['idx', 'A']).to_parquet(unique_filename)
        eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters)