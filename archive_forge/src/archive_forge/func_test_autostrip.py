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
def test_autostrip(self):
    data = '01/01/2003  , 1.3,   abcde'
    kwargs = dict(delimiter=',', dtype=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        mtest = np.genfromtxt(TextIO(data), **kwargs)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    ctrl = np.array([('01/01/2003  ', 1.3, '   abcde')], dtype=[('f0', '|S12'), ('f1', float), ('f2', '|S8')])
    assert_equal(mtest, ctrl)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
        mtest = np.genfromtxt(TextIO(data), autostrip=True, **kwargs)
        assert_(w[0].category is np.VisibleDeprecationWarning)
    ctrl = np.array([('01/01/2003', 1.3, 'abcde')], dtype=[('f0', '|S10'), ('f1', float), ('f2', '|S5')])
    assert_equal(mtest, ctrl)