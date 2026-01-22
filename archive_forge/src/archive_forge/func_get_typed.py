import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
@cachedmethod(operator.attrgetter('cache'), key=keys.typedkey)
def get_typed(self, value):
    count = self.count
    self.count += 1
    return count