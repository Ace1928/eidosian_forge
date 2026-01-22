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
def test_complex_arrays(self):
    ncols = 2
    nrows = 2
    a = np.zeros((ncols, nrows), dtype=np.complex128)
    re = np.pi
    im = np.e
    a[:] = re + 1j * im
    c = BytesIO()
    np.savetxt(c, a, fmt=' %+.3e')
    c.seek(0)
    lines = c.readlines()
    assert_equal(lines, [b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n', b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n'])
    c = BytesIO()
    np.savetxt(c, a, fmt='  %+.3e' * 2 * ncols)
    c.seek(0)
    lines = c.readlines()
    assert_equal(lines, [b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n', b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n'])
    c = BytesIO()
    np.savetxt(c, a, fmt=['(%.3e%+.3ej)'] * ncols)
    c.seek(0)
    lines = c.readlines()
    assert_equal(lines, [b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n', b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n'])