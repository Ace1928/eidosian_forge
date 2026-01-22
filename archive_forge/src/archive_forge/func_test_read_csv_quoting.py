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
@pytest.mark.parametrize('quoting', [csv.QUOTE_ALL, csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC, csv.QUOTE_NONE])
@pytest.mark.parametrize('quotechar', ['"', '_', 'd'])
@pytest.mark.parametrize('doublequote', [True, False])
@pytest.mark.parametrize('comment', [None, '#', 'x'])
def test_read_csv_quoting(self, make_csv_file, quoting, quotechar, doublequote, comment):
    use_escapechar = not doublequote and quotechar != '"' and (quoting != csv.QUOTE_NONE)
    escapechar = '\\' if use_escapechar else None
    unique_filename = make_csv_file(quoting=quoting, quotechar=quotechar, doublequote=doublequote, escapechar=escapechar, comment_col_char=comment)
    eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, quoting=quoting, quotechar=quotechar, doublequote=doublequote, escapechar=escapechar, comment=comment)