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
def test_commented_header(self):
    data = TextIO('\n#gender age weight\nM   21  72.100000\nF   35  58.330000\nM   33  21.99\n        ')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(data, names=True, dtype=None)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)], dtype=[('gender', '|S1'), ('age', int), ('weight', float)])
    assert_equal(test, ctrl)
    data = TextIO(b'\n# gender age weight\nM   21  72.100000\nF   35  58.330000\nM   33  21.99\n        ')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        test = np.genfromtxt(data, names=True, dtype=None)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    assert_equal(test, ctrl)