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
@pytest.mark.parametrize('path_type', ['object', 'directory', 'url'])
def test_read_parquet_s3(self, s3_resource, path_type, engine, s3_storage_options):
    s3_path = 's3://modin-test/modin-bugs/test_data.parquet'
    if path_type == 'object':
        import s3fs
        fs = s3fs.S3FileSystem(endpoint_url=s3_storage_options['client_kwargs']['endpoint_url'])
        with fs.open(s3_path, 'rb') as file_obj:
            eval_io('read_parquet', path=file_obj, engine=engine)
    elif path_type == 'directory':
        s3_path = 's3://modin-test/modin-bugs/test_data_dir.parquet'
        eval_io('read_parquet', path=s3_path, storage_options=s3_storage_options, engine=engine)
    else:
        eval_io('read_parquet', path=s3_path, storage_options=s3_storage_options, engine=engine)