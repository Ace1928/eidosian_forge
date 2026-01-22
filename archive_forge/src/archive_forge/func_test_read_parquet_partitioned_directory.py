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
@pytest.mark.parametrize('columns', [None, ['col1']])
@pytest.mark.parametrize('filters', [None, [], [('col1', '==', 5)], [('col1', '<=', 215), ('col2', '>=', 35)]])
@pytest.mark.parametrize('range_index_start', [0, 5000])
@pytest.mark.parametrize('range_index_step', [1, 10])
def test_read_parquet_partitioned_directory(self, tmp_path, make_parquet_file, columns, filters, range_index_start, range_index_step, engine):
    unique_filename = get_unique_filename(extension=None, data_dir=tmp_path)
    make_parquet_file(filename=unique_filename, partitioned_columns=['col1'], range_index_start=range_index_start, range_index_step=range_index_step, range_index_name='my_index')
    expected_exception = None
    if filters == [] and engine == 'pyarrow':
        expected_exception = ValueError('Malformed filters')
    eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, columns=columns, filters=filters, expected_exception=expected_exception)