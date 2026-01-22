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
@pytest.mark.parametrize('ndim', [0, 1, 2])
def test_ndmin_keyword(self, ndim: int):
    txt = '42'
    a = np.loadtxt(StringIO(txt), ndmin=ndim)
    b = np.genfromtxt(StringIO(txt), ndmin=ndim)
    assert_array_equal(a, b)