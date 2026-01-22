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
def test_buffer_null_data(pickle_module):
    null_buff = pa.foreign_buffer(address=0, size=0)
    assert null_buff.to_pybytes() == b''
    assert null_buff.address == 0
    m = memoryview(null_buff)
    assert m.tobytes() == b''
    assert pa.py_buffer(m).address != 0
    check_buffer_pickling(null_buff, pickle_module)