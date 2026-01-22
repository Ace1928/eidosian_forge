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
def test_unpack_structured(self):
    txt = TextIO('M 21 72\nF 35 58')
    dt = {'names': ('a', 'b', 'c'), 'formats': ('S1', 'i4', 'f4')}
    a, b, c = np.genfromtxt(txt, dtype=dt, unpack=True)
    assert_equal(a.dtype, np.dtype('S1'))
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(c.dtype, np.dtype('f4'))
    assert_array_equal(a, np.array([b'M', b'F']))
    assert_array_equal(b, np.array([21, 35]))
    assert_array_equal(c, np.array([72.0, 58.0]))