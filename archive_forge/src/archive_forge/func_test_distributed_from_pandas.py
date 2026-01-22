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
@pytest.mark.skipif(condition=Engine.get() != 'Ray', reason="Distributed 'from_pandas' is only available for Ray engine")
@pytest.mark.parametrize('modify_config', [{AsyncReadMode: True}], indirect=True)
def test_distributed_from_pandas(modify_config):
    pandas_df = pandas.DataFrame({f'col{i}': np.arange(200000) for i in range(64)})
    modin_df = pd.DataFrame(pandas_df)
    df_equals(modin_df, pandas_df)