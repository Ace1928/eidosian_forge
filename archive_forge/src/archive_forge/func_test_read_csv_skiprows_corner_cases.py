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
@pytest.mark.parametrize('skiprows', [[x for x in range(10)], [x + 5 for x in range(15)], [x for x in range(10) if x % 2 == 0], [x + 5 for x in range(15) if x % 2 == 0], lambda x: x % 2, lambda x: x > 20, lambda x: x < 20, lambda x: True, lambda x: x in [10, 20], lambda x: x << 10])
@pytest.mark.parametrize('header', ['infer', None, 0, 1, 150])
def test_read_csv_skiprows_corner_cases(self, skiprows, header):
    eval_io(fn_name='read_csv', check_kwargs_callable=not callable(skiprows), filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], skiprows=skiprows, header=header, dtype='str', expected_exception=False)