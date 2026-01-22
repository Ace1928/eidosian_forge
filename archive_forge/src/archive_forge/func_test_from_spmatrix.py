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
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_from_spmatrix():
    data = sparse.eye(3)
    with pytest.warns(UserWarning, match='defaulting to pandas.*'):
        modin_df = pd.DataFrame.sparse.from_spmatrix(data)
    pandas_df = pandas.DataFrame.sparse.from_spmatrix(data)
    df_equals(modin_df, pandas_df)