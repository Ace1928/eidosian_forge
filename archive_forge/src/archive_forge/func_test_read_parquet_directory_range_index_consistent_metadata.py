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
@pytest.mark.parametrize('filters', [None, [('col1', '<=', 1000000)], [('col1', '<=', 75), ('col2', '>=', 35)]])
@pytest.mark.parametrize('range_index_start', [0, 5000])
@pytest.mark.parametrize('range_index_step', [1, 10])
@pytest.mark.parametrize('range_index_name', [None, 'my_index'])
def test_read_parquet_directory_range_index_consistent_metadata(self, engine, filters, range_index_start, range_index_step, range_index_name, tmp_path):
    num_cols = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT['Small'])
    df = pandas.DataFrame({f'col{x + 1}': np.arange(0, 500) for x in range(num_cols)})
    index = pandas.RangeIndex(start=range_index_start, stop=range_index_start + len(df) * range_index_step, step=range_index_step, name=range_index_name)
    if range_index_start == 0 and range_index_step == 1 and (range_index_name is None):
        assert df.index.equals(index)
    else:
        df.index = index
    path = get_unique_filename(extension=None, data_dir=tmp_path)
    table = pa.Table.from_pandas(df)
    pyarrow.dataset.write_dataset(table, path, format='parquet', max_rows_per_group=35, max_rows_per_file=100)
    with open(os.path.join(path, '_committed_file'), 'w+') as f:
        f.write('testingtesting')
    eval_io(fn_name='read_parquet', engine=engine, path=path, filters=filters)