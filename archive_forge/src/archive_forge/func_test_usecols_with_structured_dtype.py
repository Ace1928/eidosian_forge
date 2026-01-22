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
def test_usecols_with_structured_dtype(self):
    data = TextIO('JOE 70.1 25.3\nBOB 60.5 27.9')
    names = ['stid', 'temp']
    dtypes = ['S4', 'f8']
    test = np.genfromtxt(data, usecols=(0, 2), dtype=list(zip(names, dtypes)))
    assert_equal(test['stid'], [b'JOE', b'BOB'])
    assert_equal(test['temp'], [25.3, 27.9])