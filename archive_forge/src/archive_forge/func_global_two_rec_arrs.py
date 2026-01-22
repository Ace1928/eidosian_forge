import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_two_rec_arrs(a, b, c, d):
    for i in range(len(a)):
        a[i] = rec_X[i].a
        b[i] = rec_X[i].b
        c[i] = rec_Y[i].c
        d[i] = rec_Y[i].d