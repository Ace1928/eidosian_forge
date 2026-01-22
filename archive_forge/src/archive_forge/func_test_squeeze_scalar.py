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
def test_squeeze_scalar(self):
    txt = TextIO('1')
    dt = {'names': ('a',), 'formats': ('i4',)}
    expected = np.array((1,), dtype=np.int32)
    test = np.genfromtxt(txt, dtype=dt, unpack=True)
    assert_array_equal(expected, test)
    assert_equal((), test.shape)
    assert_equal(expected.dtype, test.dtype)