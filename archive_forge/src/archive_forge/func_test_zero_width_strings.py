import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_zero_width_strings(self):
    cols = [['test'] * 3, [''] * 3]
    rec = np.rec.fromarrays(cols)
    assert_equal(rec['f0'], ['test', 'test', 'test'])
    assert_equal(rec['f1'], ['', '', ''])
    dt = np.dtype([('f0', '|S4'), ('f1', '|S')])
    rec = np.rec.fromarrays(cols, dtype=dt)
    assert_equal(rec.itemsize, 4)
    assert_equal(rec['f0'], [b'test', b'test', b'test'])
    assert_equal(rec['f1'], [b'', b'', b''])