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
def test_write_dataset_with_dataset(tempdir):
    table = pa.table({'b': ['x', 'y', 'z'], 'c': [1, 2, 3]})
    ds.write_dataset(table, tempdir, format='ipc', partitioning=['b'])
    dataset = ds.dataset(tempdir, format='ipc', partitioning=['b'])
    with tempfile.TemporaryDirectory() as tempdir2:
        ds.write_dataset(dataset, tempdir2, format='ipc', partitioning=['b'])
        load_back = ds.dataset(tempdir2, format='ipc', partitioning=['b'])
        load_back_table = load_back.to_table()
        assert dict(load_back_table.to_pydict()) == table.to_pydict()