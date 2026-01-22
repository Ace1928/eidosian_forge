import sys
from breezy import tests
from breezy.tests import features
def test_second_hash_failure(self):
    obj = self.module.SimpleSet()
    k1 = _BadSecondHash(200)
    k2 = _Hashable(200)
    obj.add(k1)
    self.assertFalse(k1._first)
    self.assertRaises(ValueError, obj.add, k2)