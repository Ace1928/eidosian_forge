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
def test_fwf_file_index_col(self, make_fwf_file):
    fwf_data = 'a       b           c          d\n' + 'id8141  360.242940  149.910199 11950.7\n' + 'id1594  444.953632  166.985655 11788.4\n' + 'id1849  364.136849  183.628767 11806.2\n' + 'id1230  413.836124  184.375703 11916.8\n' + 'id1948  502.953953  173.237159 12468.3\n'
    eval_io(fn_name='read_fwf', filepath_or_buffer=make_fwf_file(fwf_data=fwf_data), index_col='c')