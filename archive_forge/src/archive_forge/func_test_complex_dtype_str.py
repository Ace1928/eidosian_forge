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
def test_complex_dtype_str(self):
    dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), ('bottom', [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))])])
    assert_equal(str(dt), "[('top', [('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), ('bottom', [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))])]")
    dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), ('bottom', [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))])], align=True)
    assert_equal(str(dt), "{'names': ['top', 'bottom'], 'formats': [([('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)), [('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))]], 'offsets': [0, 76800], 'itemsize': 80000, 'aligned': True}")
    with np.printoptions(legacy='1.21'):
        assert_equal(str(dt), "{'names':['top','bottom'], 'formats':[([('tiles', ('>f4', (64, 64)), (1,)), ('rtile', '>f4', (64, 36))], (3,)),[('bleft', ('>f4', (8, 64)), (1,)), ('bright', '>f4', (8, 36))]], 'offsets':[0,76800], 'itemsize':80000, 'aligned':True}")
    assert_equal(np.dtype(eval(str(dt))), dt)
    dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'], 'offsets': [0, 1, 2], 'titles': ['Red pixel', 'Green pixel', 'Blue pixel']})
    assert_equal(str(dt), "[(('Red pixel', 'r'), 'u1'), (('Green pixel', 'g'), 'u1'), (('Blue pixel', 'b'), 'u1')]")
    dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'], 'formats': ['<u4', 'u1', 'u1', 'u1'], 'offsets': [0, 0, 1, 2], 'titles': ['Color', 'Red pixel', 'Green pixel', 'Blue pixel']})
    assert_equal(str(dt), "{'names': ['rgba', 'r', 'g', 'b'], 'formats': ['<u4', 'u1', 'u1', 'u1'], 'offsets': [0, 0, 1, 2], 'titles': ['Color', 'Red pixel', 'Green pixel', 'Blue pixel'], 'itemsize': 4}")
    dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'offsets': [0, 2], 'titles': ['Red pixel', 'Blue pixel']})
    assert_equal(str(dt), "{'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'offsets': [0, 2], 'titles': ['Red pixel', 'Blue pixel'], 'itemsize': 3}")
    dt = np.dtype([('a', '<m8[D]'), ('b', '<M8[us]')])
    assert_equal(str(dt), "[('a', '<m8[D]'), ('b', '<M8[us]')]")