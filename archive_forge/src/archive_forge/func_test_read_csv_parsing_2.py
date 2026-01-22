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
@pytest.mark.parametrize('header', ['infer', None, 0])
@pytest.mark.parametrize('skiprows', [2, lambda x: x % 2, lambda x: x > 25, lambda x: x > 128, np.arange(10, 50), np.arange(10, 50, 2)])
@pytest.mark.parametrize('nrows', [35, None])
@pytest.mark.parametrize('names', [[f'c{col_number}' for col_number in range(4)], [f'c{col_number}' for col_number in range(6)], None])
@pytest.mark.parametrize('encoding', ['latin1', 'windows-1251', None])
def test_read_csv_parsing_2(self, make_csv_file, request, header, skiprows, nrows, names, encoding):
    if encoding:
        unique_filename = make_csv_file(encoding=encoding)
    else:
        unique_filename = pytest.csvs_names['test_read_csv_regular']
    kwargs = {'filepath_or_buffer': unique_filename, 'header': header, 'skiprows': skiprows, 'nrows': nrows, 'names': names, 'encoding': encoding}
    if Engine.get() != 'Python':
        df = pandas.read_csv(**dict(kwargs, nrows=1))
        if df[df.columns[0]][df.index[0]] in ['c1', 'col1', 'c3', 'col3']:
            pytest.xfail('read_csv incorrect output with float data - issue #2634')
    eval_io(fn_name='read_csv', expected_exception=None, check_kwargs_callable=not callable(skiprows), **kwargs)