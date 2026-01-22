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
@pytest.mark.parametrize('dtype', [None, True])
@pytest.mark.parametrize('engine', [None, 'python', 'c'])
@pytest.mark.parametrize('converters', [None, {'col1': lambda x: np.int64(x) * 10, 'col2': pandas.to_datetime, 'col4': lambda x: x.replace(':', ';')}])
@pytest.mark.parametrize('skipfooter', [0, 10])
def test_read_csv_parsing_1(self, dtype, engine, converters, skipfooter):
    if dtype:
        dtype = {col: 'object' for col in pandas.read_csv(pytest.csvs_names['test_read_csv_regular'], nrows=1).columns}
    expected_exception = None
    if engine == 'c' and skipfooter != 0:
        expected_exception = ValueError("the 'c' engine does not support skipfooter")
    eval_io(fn_name='read_csv', expected_exception=expected_exception, check_kwargs_callable=not callable(converters), filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], dtype=dtype, engine=engine, converters=converters, skipfooter=skipfooter)