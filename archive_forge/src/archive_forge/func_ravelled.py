import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
def ravelled(A):
    try:
        return A.ravel()
    except AttributeError:
        L = []
        for idx in all_indices(A.memview.shape):
            L.append(A[idx])
        return L