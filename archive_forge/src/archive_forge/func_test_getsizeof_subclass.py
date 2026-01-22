import unittest
def test_getsizeof_subclass(self):

    class Cache(self.Cache):

        def getsizeof(self, value):
            return value
    self._test_getsizeof(Cache(maxsize=3))