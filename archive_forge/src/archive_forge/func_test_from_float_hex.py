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
def test_from_float_hex(self):
    tgt = np.logspace(-10, 10, 5).astype(np.float32)
    tgt = np.hstack((tgt, -tgt)).astype(float)
    inp = '\n'.join(map(float.hex, tgt))
    c = TextIO()
    c.write(inp)
    for dt in [float, np.float32]:
        c.seek(0)
        res = np.loadtxt(c, dtype=dt, converters=float.fromhex, encoding='latin1')
        assert_equal(res, tgt, err_msg='%s' % dt)