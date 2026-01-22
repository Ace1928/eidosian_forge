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
def test_read_xml(self):
    data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data xmlns="http://example.com">\n <row>\n   <shape>square</shape>\n   <degrees>360</degrees>\n   <sides>4.0</sides>\n </row>\n <row>\n   <shape>circle</shape>\n   <degrees>360</degrees>\n   <sides/>\n </row>\n <row>\n   <shape>triangle</shape>\n   <degrees>180</degrees>\n   <sides>3.0</sides>\n </row>\n</data>\n'
    eval_io('read_xml', path_or_buffer=data)