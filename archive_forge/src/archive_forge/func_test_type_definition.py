import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
def test_type_definition(self):
    np.random.seed(0)
    buf = []

    def print(*args):
        buf.append(args)
    alice = MyStruct('Alice', vector=np.random.random(3))

    @njit
    def make_bob():
        bob = MyStruct('unnamed', vector=np.zeros(3))
        bob.name = 'Bob'
        bob.vector = np.random.random(3)
        return bob
    bob = make_bob()
    print(f'{alice.name}: {alice.vector}')
    print(f'{bob.name}: {bob.vector}')

    @njit
    def distance(a, b):
        return np.linalg.norm(a.vector - b.vector)
    print(distance(alice, bob))
    self.assertEqual(len(buf), 3)