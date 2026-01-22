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
def test_write_dataset(tempdir):
    directory = tempdir / 'single-file'
    directory.mkdir()
    _ = _create_single_file(directory)
    dataset = ds.dataset(directory)
    target = tempdir / 'single-file-target'
    expected_files = [target / 'part-0.arrow']
    _check_dataset_roundtrip(dataset, str(target), expected_files, 'a', target)
    target = tempdir / 'single-file-target2'
    expected_files = [target / 'part-0.arrow']
    _check_dataset_roundtrip(dataset, target, expected_files, 'a', target)
    directory = tempdir / 'single-directory'
    directory.mkdir()
    _ = _create_directory_of_files(directory)
    dataset = ds.dataset(directory)
    target = tempdir / 'single-directory-target'
    expected_files = [target / 'part-0.arrow']
    _check_dataset_roundtrip(dataset, str(target), expected_files, 'a', target)