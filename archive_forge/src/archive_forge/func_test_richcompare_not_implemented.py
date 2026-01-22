import sys
from breezy import tests
from breezy.tests import features
def test_richcompare_not_implemented(self):
    obj = self.module.SimpleSet()
    k1 = _NoImplementCompare(200)
    k2 = _NoImplementCompare(200)
    self.assertLookup(200, '<null>', obj, k1)
    self.assertLookup(200, '<null>', obj, k2)
    self.assertIs(k1, obj.add(k1))
    self.assertLookup(200, k1, obj, k1)
    self.assertLookup(201, '<null>', obj, k2)
    self.assertIs(k2, obj.add(k2))
    self.assertIs(k1, obj[k1])