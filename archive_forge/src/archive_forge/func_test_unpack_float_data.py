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
def test_unpack_float_data(self):
    txt = TextIO('1,2,3\n4,5,6\n7,8,9\n0.0,1.0,2.0')
    a, b, c = np.loadtxt(txt, delimiter=',', unpack=True)
    assert_array_equal(a, np.array([1.0, 4.0, 7.0, 0.0]))
    assert_array_equal(b, np.array([2.0, 5.0, 8.0, 1.0]))
    assert_array_equal(c, np.array([3.0, 6.0, 9.0, 2.0]))