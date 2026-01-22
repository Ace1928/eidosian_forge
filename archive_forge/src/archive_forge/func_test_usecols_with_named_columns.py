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
def test_usecols_with_named_columns(self):
    ctrl = np.array([(1, 3), (4, 6)], dtype=[('a', float), ('c', float)])
    data = '1 2 3\n4 5 6'
    kwargs = dict(names='a, b, c')
    test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
    assert_equal(test, ctrl)
    test = np.genfromtxt(TextIO(data), usecols=('a', 'c'), **kwargs)
    assert_equal(test, ctrl)