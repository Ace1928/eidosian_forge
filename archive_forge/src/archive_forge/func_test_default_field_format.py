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
def test_default_field_format(self):
    data = '0, 1, 2.3\n4, 5, 6.7'
    mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=None, defaultfmt='f%02i')
    ctrl = np.array([(0, 1, 2.3), (4, 5, 6.7)], dtype=[('f00', int), ('f01', int), ('f02', float)])
    assert_equal(mtest, ctrl)