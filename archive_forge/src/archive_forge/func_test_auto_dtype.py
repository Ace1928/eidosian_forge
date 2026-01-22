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
def test_auto_dtype(self):
    data = TextIO('A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(data, dtype=None)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    control = [np.array([b'A', b'BCD']), np.array([64, 25]), np.array([75.0, 60.0]), np.array([3 + 4j, 5 + 6j]), np.array([True, False])]
    assert_equal(test.dtype.names, ['f0', 'f1', 'f2', 'f3', 'f4'])
    for i, ctrl in enumerate(control):
        assert_equal(test['f%i' % i], ctrl)