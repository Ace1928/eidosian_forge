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
@pytest.mark.orc
def test_orc_scan_options(tempdir, dataset_reader):
    from pyarrow import orc
    table = pa.table({'a': pa.array([1, 2, 3], type='int8'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    path = str(tempdir / 'test.orc')
    orc.write_table(table, path)
    dataset = ds.dataset(path, format='orc')
    result = list(dataset_reader.to_batches(dataset))
    assert len(result) == 1
    assert result[0].num_rows == 3
    assert result[0].equals(table.to_batches()[0])