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
def test_names_and_comments_none(self):
    data = TextIO('col1 col2\n 1 2\n 3 4')
    test = np.genfromtxt(data, dtype=(int, int), comments=None, names=True)
    control = np.array([(1, 2), (3, 4)], dtype=[('col1', int), ('col2', int)])
    assert_equal(test, control)