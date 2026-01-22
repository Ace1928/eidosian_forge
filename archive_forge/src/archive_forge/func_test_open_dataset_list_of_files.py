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
def test_open_dataset_list_of_files(tempdir, dataset_reader, pickle_module):
    tables, (path1, path2) = _create_directory_of_files(tempdir)
    table = pa.concat_tables(tables)
    datasets = [ds.dataset([path1, path2]), ds.dataset([str(path1), str(path2)])]
    datasets += [pickle_module.loads(pickle_module.dumps(d)) for d in datasets]
    for dataset in datasets:
        assert dataset.schema.equals(table.schema)
        result = dataset_reader.to_table(dataset)
        assert result.equals(table)