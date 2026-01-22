import random
import time
import unittest
def test_get_hit(self):
    cache = self._makeOne()
    extant = cache._data['extant'] = object()
    self.assertIs(cache.get('extant'), extant)