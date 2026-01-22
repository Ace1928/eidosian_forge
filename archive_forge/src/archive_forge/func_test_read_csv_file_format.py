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
@pytest.mark.parametrize('thousands', [None, ',', '_', ' '])
@pytest.mark.parametrize('decimal', ['.', '_'])
@pytest.mark.parametrize('lineterminator', [None, 'x', '\n'])
@pytest.mark.parametrize('escapechar', [None, 'd', 'x'])
@pytest.mark.parametrize('dialect', ['test_csv_dialect', 'use_dialect_name', None])
def test_read_csv_file_format(self, make_csv_file, thousands, decimal, lineterminator, escapechar, dialect):
    if dialect:
        test_csv_dialect_params = {'delimiter': '_', 'doublequote': False, 'escapechar': '\\', 'quotechar': 'd', 'quoting': csv.QUOTE_ALL}
        csv.register_dialect(dialect, **test_csv_dialect_params)
        if dialect != 'use_dialect_name':
            dialect = csv.get_dialect(dialect)
        unique_filename = make_csv_file(**test_csv_dialect_params)
    else:
        unique_filename = make_csv_file(thousands_separator=thousands, decimal_separator=decimal, escapechar=escapechar, lineterminator=lineterminator)
    if StorageFormat.get() == 'Hdk' and escapechar is not None and (lineterminator is None) and (thousands is None) and (decimal == '.'):
        with open(unique_filename, 'r') as f:
            if any((line.find(f',"{escapechar}') != -1 for _, line in enumerate(f))):
                pytest.xfail('Tests with this character sequence fail due to #5649')
    expected_exception = None
    if dialect is None:
        expected_exception = False
    eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, thousands=thousands, decimal=decimal, lineterminator=lineterminator, escapechar=escapechar, dialect=dialect, expected_exception=expected_exception)