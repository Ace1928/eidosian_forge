import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_repr_structured(self):
    dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), ('bottom', [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))])])
    assert_equal(repr(dt), "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), ('bottom', [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))])])")
    dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'], 'offsets': [0, 1, 2], 'titles': ['Red pixel', 'Green pixel', 'Blue pixel']}, align=True)
    assert_equal(repr(dt), "dtype([(('Red pixel', 'r'), 'u1'), (('Green pixel', 'g'), 'u1'), (('Blue pixel', 'b'), 'u1')], align=True)")