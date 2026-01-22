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
def test_uint64_type(self):
    tgt = (9223372043271415339, 9223372043271415853)
    c = TextIO()
    c.write('%s %s' % tgt)
    c.seek(0)
    res = np.loadtxt(c, dtype=np.uint64)
    assert_equal(res, tgt)