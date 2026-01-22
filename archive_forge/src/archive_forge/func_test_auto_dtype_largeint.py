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
def test_auto_dtype_largeint(self):
    data = TextIO('73786976294838206464 17179869184 1024')
    test = np.genfromtxt(data, dtype=None)
    assert_equal(test.dtype.names, ['f0', 'f1', 'f2'])
    assert_(test.dtype['f0'] == float)
    assert_(test.dtype['f1'] == np.int64)
    assert_(test.dtype['f2'] == np.int_)
    assert_allclose(test['f0'], 7.378697629483821e+19)
    assert_equal(test['f1'], 17179869184)
    assert_equal(test['f2'], 1024)