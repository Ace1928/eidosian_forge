import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
def test_shaped_dtype(self):
    c = TextIO('aaaa  1.0  8.0  1 2 3 4 5 6')
    dt = np.dtype([('name', 'S4'), ('x', float), ('y', float), ('block', int, (2, 3))])
    x = np.genfromtxt(c, dtype=dt)
    a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])], dtype=dt)
    assert_array_equal(x, a)