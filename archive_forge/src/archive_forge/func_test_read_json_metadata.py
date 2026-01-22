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
def test_read_json_metadata(self, make_json_file):
    df = pd.read_json(make_json_file(ncols=80, lines=True), lines=True, orient='records')
    parts_width_cached = df._query_compiler._modin_frame._column_widths_cache
    num_splits = len(df._query_compiler._modin_frame._partitions[0])
    parts_width_actual = [len(df._query_compiler._modin_frame._partitions[0][i].get().columns) for i in range(num_splits)]
    assert parts_width_cached == parts_width_actual