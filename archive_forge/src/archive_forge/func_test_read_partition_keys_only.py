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
def test_read_partition_keys_only(tempdir):
    BATCH_SIZE = 2 ** 15
    table = pa.table({'key': pa.repeat(0, BATCH_SIZE + 1), 'value': np.arange(BATCH_SIZE + 1)})
    pq.write_to_dataset(table[:BATCH_SIZE], tempdir / 'one', partition_cols=['key'])
    pq.write_to_dataset(table[:BATCH_SIZE + 1], tempdir / 'two', partition_cols=['key'])
    table = pq.read_table(tempdir / 'one', columns=['key'])
    assert table['key'].num_chunks == 1
    table = pq.read_table(tempdir / 'two', columns=['key', 'value'])
    assert table['key'].num_chunks == 2
    table = pq.read_table(tempdir / 'two', columns=['key'])
    assert table['key'].num_chunks == 2