import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
@guvectorize('(n)->(n)')
def my_cumsum(x, res):
    acc = 0
    for i in range(x.shape[0]):
        acc += x[i]
        res[i] = acc