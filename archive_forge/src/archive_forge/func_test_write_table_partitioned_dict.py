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
def test_write_table_partitioned_dict(tempdir):
    table = pa.table([pa.array(range(20)), pa.array(np.repeat(['a', 'b'], 10)).dictionary_encode()], names=['col', 'part'])
    partitioning = ds.partitioning(table.select(['part']).schema)
    base_dir = tempdir / 'dataset'
    ds.write_dataset(table, base_dir, format='feather', partitioning=partitioning)
    partitioning_read = ds.DirectoryPartitioning.discover(['part'], infer_dictionary=True)
    result = ds.dataset(base_dir, format='ipc', partitioning=partitioning_read).to_table()
    assert result.equals(table)