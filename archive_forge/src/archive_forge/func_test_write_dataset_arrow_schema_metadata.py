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
def test_write_dataset_arrow_schema_metadata(tempdir):
    table = pa.table({'a': [pd.Timestamp('2012-01-01', tz='Europe/Brussels')]})
    assert table['a'].type.tz == 'Europe/Brussels'
    ds.write_dataset(table, tempdir, format='parquet')
    result = pq.read_table(tempdir / 'part-0.parquet')
    assert result['a'].type.tz == 'Europe/Brussels'