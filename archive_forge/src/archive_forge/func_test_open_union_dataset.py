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
def test_open_union_dataset(tempdir, dataset_reader, pickle_module):
    _, path = _create_single_file(tempdir)
    dataset = ds.dataset(path)
    union = ds.dataset([dataset, dataset])
    assert isinstance(union, ds.UnionDataset)
    pickled = pickle_module.loads(pickle_module.dumps(union))
    assert dataset_reader.to_table(pickled) == dataset_reader.to_table(union)