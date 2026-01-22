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
def test_gft_using_filename(self):
    tgt = np.arange(6).reshape((2, 3))
    linesep = ('\n', '\r\n', '\r')
    for sep in linesep:
        data = '0 1 2' + sep + '3 4 5'
        with temppath() as name:
            with open(name, 'w') as f:
                f.write(data)
            res = np.genfromtxt(name)
        assert_array_equal(res, tgt)