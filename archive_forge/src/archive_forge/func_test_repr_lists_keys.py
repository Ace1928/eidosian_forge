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
@pytest.mark.parametrize('count, expected_repr', [(1, 'NpzFile {fname!r} with keys: arr_0'), (5, 'NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4'), (6, 'NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4...')])
def test_repr_lists_keys(self, count, expected_repr):
    a = np.array([[1, 2], [3, 4]], float)
    with temppath(suffix='.npz') as tmp:
        np.savez(tmp, *[a] * count)
        l = np.load(tmp)
        assert repr(l) == expected_repr.format(fname=tmp)
        l.close()