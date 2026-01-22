import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_objview_record(self):
    dt = np.dtype([('foo', 'i8'), ('bar', 'O')])
    r = np.zeros((1, 3), dtype=dt).view(np.recarray)
    r.foo = np.array([1, 2, 3])
    ra = np.recarray((2,), dtype=[('x', object), ('y', float), ('z', int)])
    ra[['x', 'y']]