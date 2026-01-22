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
def test_record_2(self):
    c = TextIO()
    c.write('1312 foo\n1534 bar\n4444 qux')
    c.seek(0)
    dt = [('num', np.int32), ('val', 'S3')]
    x = np.fromregex(c, '(\\d+)\\s+(...)', dt)
    a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')], dtype=dt)
    assert_array_equal(x, a)