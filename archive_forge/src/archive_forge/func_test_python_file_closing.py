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
def test_python_file_closing():
    bio = BytesIO()
    pf = pa.PythonFile(bio)
    wr = weakref.ref(pf)
    del pf
    assert wr() is None
    assert not bio.closed
    pf = pa.PythonFile(bio)
    pf.close()
    assert bio.closed