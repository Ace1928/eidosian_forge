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
def test_max_rows_larger(self):
    c = TextIO()
    c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
    c.seek(0)
    x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1, max_rows=6)
    a = np.array([[1, 2, 3, 5], [4, 5, 7, 8], [2, 1, 4, 5]], int)
    assert_array_equal(x, a)