import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
@pytest.mark.pandas
def test_dataset_project_null_column(tempdir, dataset_reader):
    df = pd.DataFrame({'col': np.array([None, None, None], dtype='object')})
    f = tempdir / 'test_dataset_project_null_column.parquet'
    df.to_parquet(f, engine='pyarrow')
    dataset = ds.dataset(f, format='parquet', schema=pa.schema([('col', pa.int64())]))
    expected = pa.table({'col': pa.array([None, None, None], pa.int64())})
    assert dataset_reader.to_table(dataset).equals(expected)