import unittest
from numba.tests.support import TestCase
def test_ex_jitclass(self):
    import numpy as np
    from numba import int32, float32
    from numba.experimental import jitclass
    spec = [('value', int32), ('array', float32[:])]

    @jitclass(spec)
    class Bag(object):

        def __init__(self, value):
            self.value = value
            self.array = np.zeros(value, dtype=np.float32)

        @property
        def size(self):
            return self.array.size

        def increment(self, val):
            for i in range(self.size):
                self.array[i] += val
            return self.array

        @staticmethod
        def add(x, y):
            return x + y
    n = 21
    mybag = Bag(n)
    self.assertTrue(isinstance(mybag, Bag))
    self.assertPreciseEqual(mybag.value, n)
    np.testing.assert_allclose(mybag.array, np.zeros(n, dtype=np.float32))
    self.assertPreciseEqual(mybag.size, n)
    np.testing.assert_allclose(mybag.increment(3), 3 * np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(mybag.increment(6), 9 * np.ones(n, dtype=np.float32))
    self.assertPreciseEqual(mybag.add(1, 1), 2)
    self.assertPreciseEqual(Bag.add(1, 2), 3)