import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_fromrecords_0len(self):
    """ Verify fromrecords works with a 0-length input """
    dtype = [('a', float), ('b', float)]
    r = np.rec.fromrecords([], dtype=dtype)
    assert_equal(r.shape, (0,))