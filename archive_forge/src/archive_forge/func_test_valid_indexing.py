import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_valid_indexing(self):
    a = np.array([[[5]]])
    a[np.array([0])]
    a[[0, 0]]
    a[:, [0, 0]]
    a[:, 0, :]
    a[:, :, :]