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
def test_header_footer(self):
    c = BytesIO()
    a = np.array([(1, 2), (3, 4)], dtype=int)
    test_header_footer = 'Test header / footer'
    np.savetxt(c, a, fmt='%1d', header=test_header_footer)
    c.seek(0)
    assert_equal(c.read(), asbytes('# ' + test_header_footer + '\n1 2\n3 4\n'))
    c = BytesIO()
    np.savetxt(c, a, fmt='%1d', footer=test_header_footer)
    c.seek(0)
    assert_equal(c.read(), asbytes('1 2\n3 4\n# ' + test_header_footer + '\n'))
    c = BytesIO()
    commentstr = '% '
    np.savetxt(c, a, fmt='%1d', header=test_header_footer, comments=commentstr)
    c.seek(0)
    assert_equal(c.read(), asbytes(commentstr + test_header_footer + '\n' + '1 2\n3 4\n'))
    c = BytesIO()
    commentstr = '% '
    np.savetxt(c, a, fmt='%1d', footer=test_header_footer, comments=commentstr)
    c.seek(0)
    assert_equal(c.read(), asbytes('1 2\n3 4\n' + commentstr + test_header_footer + '\n'))