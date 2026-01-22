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
@pytest.mark.parametrize('path_type', [str, Path])
def test_record_unicode(self, path_type):
    utf8 = b'\xcf\x96'
    with temppath() as str_path:
        path = path_type(str_path)
        with open(path, 'wb') as f:
            f.write(b'1.312 foo' + utf8 + b' \n1.534 bar\n4.444 qux')
        dt = [('num', np.float64), ('val', 'U4')]
        x = np.fromregex(path, '(?u)([0-9.]+)\\s+(\\w+)', dt, encoding='UTF-8')
        a = np.array([(1.312, 'foo' + utf8.decode('UTF-8')), (1.534, 'bar'), (4.444, 'qux')], dtype=dt)
        assert_array_equal(x, a)
        regexp = re.compile('([0-9.]+)\\s+(\\w+)', re.UNICODE)
        x = np.fromregex(path, regexp, dt, encoding='UTF-8')
        assert_array_equal(x, a)