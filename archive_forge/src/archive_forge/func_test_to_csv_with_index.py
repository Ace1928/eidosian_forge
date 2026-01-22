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
def test_to_csv_with_index(self, tmp_path):
    cols = 100
    arows = 20000
    keyrange = 100
    values = np.vstack([np.random.choice(keyrange, size=arows), np.random.normal(size=(cols, arows))]).transpose()
    modin_df = pd.DataFrame(values, columns=['key'] + ['avalue' + str(i) for i in range(1, 1 + cols)]).set_index('key')
    pandas_df = pandas.DataFrame(values, columns=['key'] + ['avalue' + str(i) for i in range(1, 1 + cols)]).set_index('key')
    eval_to_csv_file(tmp_path, modin_df, pandas_df, 'csv')