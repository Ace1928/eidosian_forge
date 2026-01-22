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
@pytest.mark.parametrize(['skip', 'data'], [(1, ['ignored\n', '1,2\n', '\n', '3,4\n']), (1, ['ignored', '1,2', '', '3,4']), (1, StringIO('ignored\n1,2\n\n3,4')), (0, ['-1,0\n', '1,2\n', '\n', '3,4\n']), (0, ['-1,0', '1,2', '', '3,4']), (0, StringIO('-1,0\n1,2\n\n3,4'))])
def test_max_rows_empty_lines(self, skip, data):
    with pytest.warns(UserWarning, match=f'Input line 3.*max_rows={3 - skip}'):
        res = np.loadtxt(data, dtype=int, skiprows=skip, delimiter=',', max_rows=3 - skip)
        assert_array_equal(res, [[-1, 0], [1, 2], [3, 4]][skip:])
    if isinstance(data, StringIO):
        data.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning):
            np.loadtxt(data, dtype=int, skiprows=skip, delimiter=',', max_rows=3 - skip)