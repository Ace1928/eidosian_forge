import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_from_obj(self):
    count = 10
    a = np.zeros(count, dtype='O')
    b = np.zeros(count, dtype='f8')
    c = np.zeros(count, dtype='f8')
    for i in range(len(a)):
        a[i] = list(range(1, 10))
    mine = np.rec.fromarrays([a, b, c], names='date,data1,data2')
    for i in range(len(a)):
        assert_(mine.date[i] == list(range(1, 10)))
        assert_(mine.data1[i] == 0.0)
        assert_(mine.data2[i] == 0.0)