import random
import time
import unittest
def test_get_miss_no_default(self):
    cache = self._makeOne()
    self.assertIsNone(cache.get('nonesuch'))