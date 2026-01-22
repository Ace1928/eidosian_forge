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
def test_fwf_file_chunksize(self, make_fwf_file):
    unique_filename = make_fwf_file()
    rdf_reader = pd.read_fwf(unique_filename, chunksize=5)
    pd_reader = pandas.read_fwf(unique_filename, chunksize=5)
    for modin_df, pd_df in zip(rdf_reader, pd_reader):
        df_equals(modin_df, pd_df)
    rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
    pd_reader = pandas.read_fwf(unique_filename, chunksize=1)
    modin_df = rdf_reader.get_chunk(1)
    pd_df = pd_reader.get_chunk(1)
    df_equals(modin_df, pd_df)
    rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
    pd_reader = pandas.read_fwf(unique_filename, chunksize=1)
    modin_df = rdf_reader.read()
    pd_df = pd_reader.read()
    df_equals(modin_df, pd_df)