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
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
@pytest.mark.parametrize('encoding_errors', ['strict', 'ignore'])
@pytest.mark.parametrize('parse_dates', [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()])
@pytest.mark.parametrize('index_col', [None, 0, 5])
@pytest.mark.parametrize('header', ['infer', 0])
@pytest.mark.parametrize('names', [None, ['timestamp', 'year', 'month', 'date', 'symbol', 'high', 'low', 'open', 'close', 'spread', 'volume']])
@pytest.mark.exclude_in_sanity
def test_read_csv_parse_dates(self, names, header, index_col, parse_dates, encoding, encoding_errors, request):
    if names is not None and header == 'infer':
        pytest.xfail('read_csv with Ray engine works incorrectly with date data and names parameter provided - issue #2509')
    expected_exception = None
    if 'nonexistent_int_column' in request.node.callspec.id:
        expected_exception = IndexError('list index out of range')
    elif 'nonexistent_string_column' in request.node.callspec.id:
        expected_exception = ValueError("Missing column provided to 'parse_dates': 'z'")
    if StorageFormat.get() == 'Hdk' and 'names1-0-None-nonexistent_string_column-strict-None' in request.node.callspec.id:
        expected_exception = False
    eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer=time_parsing_csv_path, names=names, header=header, index_col=index_col, parse_dates=parse_dates, encoding=encoding, encoding_errors=encoding_errors)