import sys
from breezy import tests
from breezy.tests import features
def test__lookup_collision(self):
    obj = self.module.SimpleSet()
    k1 = _Hashable(643)
    k2 = _Hashable(643 + 1024)
    self.assertLookup(643, '<null>', obj, k1)
    self.assertLookup(643, '<null>', obj, k2)
    obj.add(k1)
    self.assertLookup(643, k1, obj, k1)
    self.assertLookup(644, '<null>', obj, k2)