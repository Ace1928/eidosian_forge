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
def check_large_zip(memoryerror_raised):
    memoryerror_raised.value = False
    try:
        test_data = np.asarray([np.random.rand(np.random.randint(50, 100), 4) for i in range(800000)], dtype=object)
        with tempdir() as tmpdir:
            np.savez(os.path.join(tmpdir, 'test.npz'), test_data=test_data)
    except MemoryError:
        memoryerror_raised.value = True
        raise