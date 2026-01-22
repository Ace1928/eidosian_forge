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
def test_custom_writer(self):

    class CustomWriter(list):

        def write(self, text):
            self.extend(text.split(b'\n'))
    w = CustomWriter()
    a = np.array([(1, 2), (3, 4)])
    np.savetxt(w, a)
    b = np.loadtxt(w)
    assert_array_equal(a, b)