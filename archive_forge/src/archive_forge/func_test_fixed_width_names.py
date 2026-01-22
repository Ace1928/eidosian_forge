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
def test_fixed_width_names(self):
    data = '    A    B   C\n    0    1 2.3\n   45   67   9.'
    kwargs = dict(delimiter=(5, 5, 4), names=True, dtype=None)
    ctrl = np.array([(0, 1, 2.3), (45, 67, 9.0)], dtype=[('A', int), ('B', int), ('C', float)])
    test = np.genfromtxt(TextIO(data), **kwargs)
    assert_equal(test, ctrl)
    kwargs = dict(delimiter=5, names=True, dtype=None)
    ctrl = np.array([(0, 1, 2.3), (45, 67, 9.0)], dtype=[('A', int), ('B', int), ('C', float)])
    test = np.genfromtxt(TextIO(data), **kwargs)
    assert_equal(test, ctrl)