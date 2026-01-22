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
def test_dtype_with_object(self):
    data = ' 1; 2001-01-01\n                   2; 2002-01-31 '
    ndtype = [('idx', int), ('code', object)]
    func = lambda s: strptime(s.strip(), '%Y-%m-%d')
    converters = {1: func}
    test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)
    control = np.array([(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))], dtype=ndtype)
    assert_equal(test, control)
    ndtype = [('nest', [('idx', int), ('code', object)])]
    with assert_raises_regex(NotImplementedError, 'Nested fields.* not supported.*'):
        test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)
    ndtype = [('idx', int), ('code', object), ('nest', [])]
    with assert_raises_regex(NotImplementedError, 'Nested fields.* not supported.*'):
        test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)