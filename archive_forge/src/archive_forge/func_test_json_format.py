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
@pytest.mark.pandas
def test_json_format(tempdir, dataset_reader):
    table = pa.table({'a': pa.array([1, 2, 3], type='int64'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    path = str(tempdir / 'test.json')
    out = table.to_pandas().to_json(orient='records')[1:-1].replace('},{', '}\n{')
    with open(path, 'w') as f:
        f.write(out)
    dataset = ds.dataset(path, format=ds.JsonFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)
    assert_dataset_fragment_convenience_methods(dataset)
    dataset = ds.dataset(path, format='json')
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)