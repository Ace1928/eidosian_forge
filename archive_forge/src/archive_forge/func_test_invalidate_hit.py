import random
import time
import unittest
def test_invalidate_hit(self):
    cache = self._makeOne()
    extant = cache._data['extant'] = object()
    cache.invalidate('extant')
    self.assertIsNone(cache.get('extant'))