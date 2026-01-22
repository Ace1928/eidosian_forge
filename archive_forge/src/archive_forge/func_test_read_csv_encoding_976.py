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
@pytest.mark.parametrize('pathlike', [False, True])
def test_read_csv_encoding_976(self, pathlike):
    file_name = 'modin/tests/pandas/data/issue_976.csv'
    if pathlike:
        file_name = Path(file_name)
    names = [str(i) for i in range(11)]
    kwargs = {'sep': ';', 'names': names, 'encoding': 'windows-1251'}
    df1 = pd.read_csv(file_name, **kwargs)
    df2 = pandas.read_csv(file_name, **kwargs)
    df1 = df1.drop(['4', '5'], axis=1)
    df2 = df2.drop(['4', '5'], axis=1)
    df_equals(df1, df2)