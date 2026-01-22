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
def test_open_dataset_non_existing_file():
    with pytest.raises(FileNotFoundError):
        ds.dataset('i-am-not-existing.arrow', format='ipc')
    with pytest.raises(pa.ArrowInvalid, match='cannot be relative'):
        ds.dataset('file:i-am-not-existing.arrow', format='ipc')