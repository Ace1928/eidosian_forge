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
def test_multifield_view(self):
    a = np.ones(1, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
    v = a[['x', 'z']]
    with temppath(suffix='.npy') as path:
        path = Path(path)
        np.save(path, v)
        data = np.load(path)
        assert_array_equal(data, v)