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
@pytest.mark.parametrize('iterator', [True, False])
def test_read_csv_iteration(self, iterator):
    filename = pytest.csvs_names['test_read_csv_regular']
    rdf_reader = pd.read_csv(filename, chunksize=500, iterator=iterator)
    pd_reader = pandas.read_csv(filename, chunksize=500, iterator=iterator)
    for modin_df, pd_df in zip(rdf_reader, pd_reader):
        df_equals(modin_df, pd_df)
    rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
    pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)
    modin_df = rdf_reader.get_chunk(1)
    pd_df = pd_reader.get_chunk(1)
    df_equals(modin_df, pd_df)
    rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
    pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)
    modin_df = rdf_reader.read()
    pd_df = pd_reader.read()
    df_equals(modin_df, pd_df)
    if iterator:
        rdf_reader = pd.read_csv(filename, iterator=iterator)
        pd_reader = pandas.read_csv(filename, iterator=iterator)
        modin_df = rdf_reader.read()
        pd_df = pd_reader.read()
        df_equals(modin_df, pd_df)