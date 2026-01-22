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
def test_converters_cornercases2(self):
    converter = {'date': lambda s: np.datetime64(strptime(s, '%Y-%m-%d %H:%M:%SZ'))}
    data = TextIO('2009-02-03 12:00:00Z, 72214.0')
    test = np.genfromtxt(data, delimiter=',', dtype=None, names=['date', 'stid'], converters=converter)
    control = np.array((datetime(2009, 2, 3), 72214.0), dtype=[('date', 'datetime64[us]'), ('stid', float)])
    assert_equal(test, control)