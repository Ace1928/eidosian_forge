import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
def test_overload_method(self):
    from numba.core.extending import overload_method
    from numba.core.errors import TypingError

    @overload_method(MyStructType, 'distance')
    def ol_distance(self, other):
        if not isinstance(other, MyStructType):
            raise TypingError(f'*other* must be a {MyStructType}; got {other}')

        def impl(self, other):
            return np.linalg.norm(self.vector - other.vector)
        return impl

    @njit
    def test():
        alice = MyStruct('Alice', vector=np.random.random(3))
        bob = MyStruct('Bob', vector=np.random.random(3))
        return alice.distance(bob)
    self.assertIsInstance(test(), float)