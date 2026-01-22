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
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_load_refcount():
    f = BytesIO()
    np.savez(f, [1, 2, 3])
    f.seek(0)
    with assert_no_gc_cycles():
        np.load(f)
    f.seek(0)
    dt = [('a', 'u1', 2), ('b', 'u1', 2)]
    with assert_no_gc_cycles():
        x = np.loadtxt(TextIO('0 1 2 3'), dtype=dt)
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))