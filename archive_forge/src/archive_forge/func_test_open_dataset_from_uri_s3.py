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
@pytest.mark.s3
def test_open_dataset_from_uri_s3(s3_example_simple, dataset_reader):
    table, path, fs, uri, _, _, _, _ = s3_example_simple
    dataset = ds.dataset(uri, format='parquet')
    assert dataset_reader.to_table(dataset).equals(table)
    dataset = ds.dataset(path, format='parquet', filesystem=fs)
    assert dataset_reader.to_table(dataset).equals(table)