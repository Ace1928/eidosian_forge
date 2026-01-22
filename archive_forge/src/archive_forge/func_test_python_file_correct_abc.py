import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_python_file_correct_abc():
    with pa.PythonFile(BytesIO(b''), mode='r') as f:
        assert isinstance(f, BufferedIOBase)
        assert isinstance(f, IOBase)