import sys
from breezy import tests
from breezy.tests import features
def test__iter__(self):
    obj = self.module.SimpleSet()
    k1 = ('1',)
    k2 = ('1', '2')
    k3 = ('3', '4')
    obj.add(k1)
    obj.add(k2)
    obj.add(k3)
    all = set()
    for key in obj:
        all.add(key)
    self.assertEqual(sorted([k1, k2, k3]), sorted(all))
    iterator = iter(obj)
    self.assertIn(next(iterator), all)
    obj.add(('foo',))
    self.assertRaises(RuntimeError, next, iterator)
    obj.discard(k2)
    self.assertRaises(RuntimeError, next, iterator)