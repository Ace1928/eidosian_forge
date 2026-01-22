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
def test_inconsistent_dtype(self):
    data = ['1, 1, 1, 1, -1.1'] * 50
    mdata = TextIO('\n'.join(data))
    converters = {4: lambda x: '(%s)' % x.decode()}
    kwargs = dict(delimiter=',', converters=converters, dtype=[(_, int) for _ in 'abcde'])
    assert_raises(ValueError, np.genfromtxt, mdata, **kwargs)