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
def test_construct_dataset_with_invalid_schema():
    empty = ds.dataset([], format='ipc', schema=pa.schema([('a', pa.int64()), ('a', pa.string())]))
    with pytest.raises(ValueError, match='Multiple matches for .*a.* in '):
        empty.to_table()