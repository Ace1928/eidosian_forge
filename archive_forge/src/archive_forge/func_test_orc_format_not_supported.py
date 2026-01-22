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
def test_orc_format_not_supported():
    try:
        from pyarrow.dataset import OrcFileFormat
    except ImportError:
        with pytest.raises(ValueError, match='not built with support for the ORC file'):
            ds.dataset('.', format='orc')