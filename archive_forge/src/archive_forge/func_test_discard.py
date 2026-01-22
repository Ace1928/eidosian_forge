import sys
from breezy import tests
from breezy.tests import features
def test_discard(self):
    obj = self.module.SimpleSet()
    k1 = tuple(['foo'])
    k2 = tuple(['foo'])
    k3 = tuple(['bar'])
    self.assertRefcount(1, k1)
    self.assertRefcount(1, k2)
    self.assertRefcount(1, k3)
    obj.add(k1)
    self.assertRefcount(2, k1)
    self.assertEqual(0, obj.discard(k3))
    self.assertRefcount(1, k3)
    obj.add(k3)
    self.assertRefcount(2, k3)
    self.assertEqual(1, obj.discard(k3))
    self.assertRefcount(1, k3)