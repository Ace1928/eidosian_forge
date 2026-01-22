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
def test_incomplete_names(self):
    data = 'A,,C\n0,1,2\n3,4,5'
    kwargs = dict(delimiter=',', names=True)
    ctrl = np.array([(0, 1, 2), (3, 4, 5)], dtype=[(_, int) for _ in ('A', 'f0', 'C')])
    test = np.genfromtxt(TextIO(data), dtype=None, **kwargs)
    assert_equal(test, ctrl)
    ctrl = np.array([(0, 1, 2), (3, 4, 5)], dtype=[(_, float) for _ in ('A', 'f0', 'C')])
    test = np.genfromtxt(TextIO(data), **kwargs)