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
def test_to_parquet_s3(self, s3_resource, engine, s3_storage_options):
    modin_path = 's3://modin-test/modin-dir/modin_df.parquet'
    mdf, pdf = create_test_dfs(utils_test_data['int_data'])
    pdf.to_parquet('s3://modin-test/pandas-dir/pandas_df.parquet', engine=engine, storage_options=s3_storage_options)
    mdf.to_parquet(modin_path, engine=engine, storage_options=s3_storage_options)
    df_equals(pandas.read_parquet('s3://modin-test/pandas-dir/pandas_df.parquet', storage_options=s3_storage_options), pd.read_parquet(modin_path, storage_options=s3_storage_options))
    assert not os.path.isdir(modin_path)