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
def test_dataset_union(multisourcefs):
    child = ds.FileSystemDatasetFactory(multisourcefs, fs.FileSelector('/plain'), format=ds.ParquetFileFormat())
    factory = ds.UnionDatasetFactory([child])
    assert len(factory.inspect_schemas()) == 1
    assert all((isinstance(s, pa.Schema) for s in factory.inspect_schemas()))
    assert factory.inspect_schemas()[0].equals(child.inspect())
    assert factory.inspect().equals(child.inspect())
    assert isinstance(factory.finish(), ds.Dataset)