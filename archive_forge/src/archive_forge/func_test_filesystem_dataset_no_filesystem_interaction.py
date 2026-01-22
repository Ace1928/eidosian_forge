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
def test_filesystem_dataset_no_filesystem_interaction(dataset_reader):
    schema = pa.schema([pa.field('f1', pa.int64())])
    file_format = ds.IpcFileFormat()
    paths = ['nonexistingfile.arrow']
    dataset = ds.FileSystemDataset.from_paths(paths, schema=schema, format=file_format, filesystem=fs.LocalFileSystem())
    dataset.get_fragments()
    with pytest.raises(FileNotFoundError):
        dataset_reader.to_table(dataset)